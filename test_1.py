import pybullet as p
import pybullet_data
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import math
import matplotlib.pyplot as plt
import os # For path joining

# --- Simulation Environment ---
class MazeEnv:
    def __init__(self, urdf_root='/home/beta_frame/Documents/5756_project/asset/', render=False):
        self.urdf_root = urdf_root
        self._render = render
        self._physics_client_id = p.connect(p.GUI if self._render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Adjust search path if your URDFs/meshes are elsewhere
        p.loadURDF("plane.urdf")
        p.setAdditionalSearchPath(os.path.abspath(self.urdf_root))


        # Environment parameters
        self.start_pos = [-1.5, -1.5, 0.07] # Start slightly above ground
        self.start_orn_q = p.getQuaternionFromEuler([0, 0, 0])
        self.goal_pos = [1.5, 1.5] # Goal X, Y
        self.goal_threshold = 0.3 # Meters
        self.max_steps_per_episode = 1000
        self.time_step = 1. / 240. # PyBullet default time step

        # Robot parameters
        self.robot_urdf_path = os.path.join(self.urdf_root, "mentorpi_description/urdf/ack.xacro")
        self.maze_urdf_path = os.path.join(self.urdf_root, "maze/maze_generator_ros/maze/model.urdf")
        self.wheel_joint_names = ["wheel_lf_Joint", "wheel_rf_Joint", "wheel_lb_Joint", "wheel_rb_Joint"]
        self.lidar_link_name = "lidar_frame"
        self.base_link_name = "base_link"
        self.num_lidar_rays = 36 # Number of Lidar rays
        self.lidar_range = 3.0 # Max Lidar range
        self.robot_id = None
        self.maze_id = None
        self.wheel_joint_indices = []
        self.lidar_link_index = -1
        self.base_link_index = -1

        # Action space (discrete)
        # 0: Forward, 1: Turn Left, 2: Turn Right
        self.action_space_dim = 3
        self.linear_velocity = 5 # rad/s for wheels
        self.angular_velocity = 10 # rad/s difference for turning

        # State space
        # Lidar (num_lidar_rays) + relative goal distance + relative goal angle
        self.state_space_dim = self.num_lidar_rays + 2

        self.current_step = 0
        self.episode_reward = 0

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

        self.load_models()


    def load_models(self):
        # Load maze - fixed base
        try:
            self.maze_id = p.loadURDF(self.maze_urdf_path, [0, 0, 0], useFixedBase=1, physicsClientId=self._physics_client_id)
            print(f"Maze URDF loaded successfully from {self.maze_urdf_path}")
            
        except Exception as e:
            print(f"Error loading maze URDF from {self.maze_urdf_path}: {e}")
            raise

        # Load robot
        try:
            self.robot_id = p.loadURDF(self.robot_urdf_path, self.start_pos, self.start_orn_q, physicsClientId=self._physics_client_id)
            print(f"Robot URDF loaded successfully from {self.robot_urdf_path}")
        except Exception as e:
            print(f"Error loading robot URDF from {self.robot_urdf_path}: {e}")
            print("Ensure mesh paths in ack.urdf are correct and relative to the script or PyBullet's search path.")
            raise

        # Get joint and link info
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self._physics_client_id)
        print(f"Robot has {num_joints} joints.")
        found_lidar = False
        found_base = False
        found_wheels = 0
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self._physics_client_id)
            joint_name = info[1].decode('utf-8')
            link_name = info[12].decode('utf-8')
            #print(f"Joint {i}: {joint_name}, Link: {link_name}") # Debug print

            if joint_name in self.wheel_joint_names:
                self.wheel_joint_indices.append(i)
                found_wheels += 1
                # Set friction for wheels if needed
                # p.changeDynamics(self.robot_id, i, lateralFriction=0.8)

            if link_name == self.lidar_link_name:
                self.lidar_link_index = i # Note: This gets the *joint* index whose *child link* is the lidar frame
                found_lidar = True

            if link_name == self.base_link_name:
                 self.base_link_index = i # Note: This gets the *joint* index whose *child link* is the base_link
                 found_base = True


        if len(self.wheel_joint_indices) != 4:
             print(f"Warning: Expected 4 wheel joints, found {len(self.wheel_joint_indices)}. Check names: {self.wheel_joint_names}")
        if not found_lidar:
             print(f"Warning: Lidar link '{self.lidar_link_name}' not found. Check link name in URDF.")
             # Fallback: use base link index if lidar frame isn't explicitly found as a link name in joint info
             if found_base:
                 print("Attempting to use base_link index for Lidar instead.")
                 self.lidar_link_index = self.base_link_index # Use base link if lidar link not found
             else:
                 print("Error: Neither lidar_frame nor base_link found. Cannot simulate Lidar.")
                 self.lidar_link_index = -1 # Default to base if needed, though results might be off
        else:
            print(f"Lidar link '{self.lidar_link_name}' found at index {self.lidar_link_index}.")

        print(f"Wheel joint indices: {self.wheel_joint_indices}")


    def get_lidar_scan(self):
        if self.lidar_link_index == -1:
             print("Error: Lidar link index not set.")
             return np.zeros(self.num_lidar_rays) # Return zeros if no lidar link

        lidar_state = p.getLinkState(self.robot_id, self.lidar_link_index, computeForwardKinematics=True, physicsClientId=self._physics_client_id)
        lidar_pos = lidar_state[0]
        lidar_orn = lidar_state[1] # Quaternion

        # Prepare ray casting
        ray_from = [lidar_pos] * self.num_lidar_rays
        ray_to = []
        lidar_basis = p.getMatrixFromQuaternion(lidar_orn)
        # Forward vector is the first column of the rotation matrix
        # Left vector is the second column
        # Up vector is the third column
        forward_vec = np.array([lidar_basis[0], lidar_basis[3], lidar_basis[6]])
        up_vec = np.array([lidar_basis[2], lidar_basis[5], lidar_basis[8]]) # Z-axis in link frame

        angles = np.linspace(-np.pi, np.pi, self.num_lidar_rays, endpoint=False) # 360 degrees scan

        for angle in angles:
            # Rotate the forward vector around the link's up vector
            rot_quat = p.getQuaternionFromAxisAngle(up_vec, angle)
            rot_mat = p.getMatrixFromQuaternion(rot_quat)
            rotated_forward = np.dot(np.array(rot_mat).reshape(3, 3), forward_vec)

            end_pos = np.array(lidar_pos) + self.lidar_range * rotated_forward
            ray_to.append(end_pos.tolist())

        results = p.rayTestBatch(ray_from, ray_to, physicsClientId=self._physics_client_id)

        # Process results
        distances = []
        for result in results:
            hit_fraction = result[2]
            distances.append(hit_fraction * self.lidar_range)

        # Normalize distances
        normalized_distances = np.array(distances) / self.lidar_range
        return normalized_distances


    def get_state(self):
        # Lidar Scan
        # lidar_scan = self.get_lidar_scan()

        # Robot Pose
        base_pos, base_orn_q = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._physics_client_id)
        base_eul = p.getEulerFromQuaternion(base_orn_q) # ZYX roll, pitch, yaw
        robot_x, robot_y, _ = base_pos
        robot_yaw = base_eul[2]

        # Relative Goal Info
        goal_dx = self.goal_pos[0] - robot_x
        goal_dy = self.goal_pos[1] - robot_y
        distance_to_goal = math.sqrt(goal_dx**2 + goal_dy**2)

        # Angle to goal relative to robot's current orientation
        angle_to_goal = math.atan2(goal_dy, goal_dx)
        relative_angle = angle_to_goal - robot_yaw
        # Normalize angle to [-pi, pi]
        while relative_angle > math.pi: relative_angle -= 2 * math.pi
        while relative_angle < -math.pi: relative_angle += 2 * math.pi

        # Normalize relative info (optional but good practice)
        norm_dist_to_goal = distance_to_goal / 4.0 # Normalize by approx maze diagonal
        norm_rel_angle = relative_angle / math.pi

        # Combine state
        state = np.array([norm_dist_to_goal, norm_rel_angle])
        return state

    def calculate_reward(self, prev_state, done, collision, reached_goal):
        reward = 0

        # Penalty for collision
        if collision:
            reward -= 100.0
            #print("Collision!")
            return reward # End reward calculation on collision

        # Large reward for reaching the goal
        if reached_goal:
            reward += 200.0
            #print("Reached Goal!")
            return reward

        # Reward for getting closer to the goal
        prev_dist = prev_state[-2] * 4.0 # Denormalize
        current_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._physics_client_id)
        current_dist = math.sqrt((self.goal_pos[0] - current_pos[0])**2 + (self.goal_pos[1] - current_pos[1])**2)
        distance_delta = prev_dist - current_dist
        reward += distance_delta * 10.0 # Scale reward for distance change

        # Small penalty per step to encourage efficiency
        reward -= 0.1

        return reward

    def check_collision(self):
        contact_points = p.getContactPoints(bodyA=self.robot_id, bodyB=self.maze_id, physicsClientId=self._physics_client_id)
        return len(contact_points) > 0

    def check_goal_reached(self):
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._physics_client_id)
        distance = math.sqrt((self.goal_pos[0] - base_pos[0])**2 + (self.goal_pos[1] - base_pos[1])**2)
        return distance < self.goal_threshold

    def reset(self):
        # Reset robot pose
        p.resetBasePositionAndOrientation(self.robot_id, self.start_pos, self.start_orn_q, physicsClientId=self._physics_client_id)
        # Reset robot velocities
        p.resetBaseVelocity(self.robot_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0], physicsClientId=self._physics_client_id)
        # Ensure joints are reset too
        for joint_index in self.wheel_joint_indices:
             p.resetJointState(self.robot_id, joint_index, targetValue=0, targetVelocity=0, physicsClientId=self._physics_client_id)

        self.current_step = 0
        self.episode_reward = 0

        # Needed to get first state correctly after reset
        p.stepSimulation(physicsClientId=self._physics_client_id)

        return self.get_state()

    def step(self, action):
        prev_state = self.get_state() # Get state before applying action

        # --- Apply Action ---
        # Indices based on typical Ackermann/Differential drive mapping
        # wheel_lf_Joint, wheel_rf_Joint, wheel_lb_Joint, wheel_rb_Joint
        left_speed = 0
        right_speed = 0

        if action == 0: # Forward
            left_speed = self.linear_velocity
            right_speed = self.linear_velocity
        elif action == 1: # Turn Left
            left_speed = -self.angular_velocity / 2
            right_speed = self.angular_velocity / 2
        elif action == 2: # Turn Right
            left_speed = self.angular_velocity / 2
            right_speed = -self.angular_velocity / 2
        # else: # Optional: Stop or other actions
        #     left_speed = 0
        #     right_speed = 0

        # Apply velocity to wheel joints
        # Assuming [LF, RF, LB, RB] order from your URDF and self.wheel_joint_names
        if len(self.wheel_joint_indices) == 4:
            p.setJointMotorControl2(self.robot_id, self.wheel_joint_indices[2], p.VELOCITY_CONTROL, targetVelocity=left_speed, force=50.0, physicsClientId=self._physics_client_id)
            p.setJointMotorControl2(self.robot_id, self.wheel_joint_indices[3], p.VELOCITY_CONTROL, targetVelocity=right_speed, force=50.0, physicsClientId=self._physics_client_id)
        else:
            print("Error: Incorrect number of wheel joints found. Cannot apply action.")


        # --- Step Simulation ---
        p.stepSimulation(physicsClientId=self._physics_client_id)
        if self._render:
            time.sleep(self.time_step) # Slow down for visualization

        self.current_step += 1

        # --- Get New State and Check Conditions ---
        next_state = self.get_state()
        collision = self.check_collision()
        reached_goal = self.check_goal_reached()

        # Determine if episode is done
        done = False
        if reached_goal:
            done = True
            print(f"Episode finished: Reached Goal in {self.current_step} steps.")
        elif collision:
            done = True
            print(f"Episode finished: Collision after {self.current_step} steps.")
        elif self.current_step >= self.max_steps_per_episode:
            done = True
            print(f"Episode finished: Max steps ({self.max_steps_per_episode}) reached.")

        # --- Calculate Reward ---
        reward = self.calculate_reward(prev_state, done, collision, reached_goal)
        self.episode_reward += reward

        info = {'collision': collision, 'reached_goal': reached_goal} # Optional info dictionary

        return next_state, reward, done, info


    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect(physicsClientId=self._physics_client_id)
            self._physics_client_id = -1

# --- DQN Agent ---

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        # Convert to numpy arrays before converting to tensors
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array(action, dtype=np.int64) # Actions are indices
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.float32) # Use float for multiplication later
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim, replay_buffer_capacity=10000, batch_size=64, gamma=0.99, lr=1e-3, target_update_freq=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval() # Target network is not trained directly

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.loss_fn = nn.MSELoss()


    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim) # Explore
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) # Add batch dimension
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item() # Exploit

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None # Not enough samples yet

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device) # Shape [batch_size, 1]
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device) # Shape [batch_size, 1]
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device) # Shape [batch_size, 1]

        # Get current Q values for the actions taken
        # q_network output shape: [batch_size, action_dim]
        # gather needs index shape: [batch_size, 1]
        current_q_values = self.q_network(states_t).gather(1, actions_t)

        # Get next Q values from target network
        with torch.no_grad():
            next_q_values_target = self.target_network(next_states_t).max(1)[0].unsqueeze(1) # Shape [batch_size, 1]

        # Compute target Q values: R + gamma * max_a' Q_target(s', a')
        # If done is true (1.0), the target is just the reward
        target_q_values = rewards_t + (self.gamma * next_q_values_target * (1 - dones_t))

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()


    def update_target_network(self):
        #print("Updating target network...")
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
         try:
            self.q_network.load_state_dict(torch.load(path, map_location=self.device))
            self.target_network.load_state_dict(self.q_network.state_dict()) # Sync target network
            self.q_network.eval() # Set to evaluation mode if not training further
            self.target_network.eval()
            print(f"Model loaded from {path}")
         except FileNotFoundError:
             print(f"Error: Model file not found at {path}. Starting from scratch.")
         except Exception as e:
             print(f"Error loading model from {path}: {e}. Starting from scratch.")


# --- Training Loop ---
if __name__ == "__main__":
    # --- Hyperparameters ---
    NUM_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 100000 # Redundant with env setting, but useful here
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    REPLAY_BUFFER_CAPACITY = 50000
    BATCH_SIZE = 128
    TARGET_UPDATE_FREQ = 200 # Update target network every N training steps
    TRAIN_EVERY_N_STEPS = 4 # Train the network every N simulation steps
    MODEL_SAVE_PATH = "dqn_maze_robot.pth"
    # Directory containing ack.urdf, simple_maze.urdf and meshes folder
    RENDER_TRAINING = True # Set to True to watch training (slower)

    # --- Initialization ---
    env = MazeEnv(render=RENDER_TRAINING)
    env.max_steps_per_episode = MAX_STEPS_PER_EPISODE # Sync env parameter

    agent = DQNAgent(
        state_dim=env.state_space_dim,
        action_dim=env.action_space_dim,
        replay_buffer_capacity=REPLAY_BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        lr=LEARNING_RATE,
        target_update_freq=TARGET_UPDATE_FREQ
    )

    # Optional: Load pre-trained model
    # agent.load_model(MODEL_SAVE_PATH)

    epsilon = EPSILON_START
    episode_rewards = []
    episode_losses = []
    total_steps = 0

    # --- Training ---
    try:
        for episode in range(NUM_EPISODES):
            state = env.reset()
            episode_reward = 0
            step_losses = []

            for step in range(MAX_STEPS_PER_EPISODE):
                action = agent.select_action(state, epsilon)
                next_state, reward, done, info = env.step(action)

                agent.store_transition(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                total_steps += 1

                # Train the agent
                if total_steps % TRAIN_EVERY_N_STEPS == 0 and len(agent.replay_buffer) >= BATCH_SIZE:
                     loss = agent.train()
                     if loss is not None:
                         step_losses.append(loss)

                if RENDER_TRAINING and step % 50 == 0: # Add небольшой pause for better viewing
                    pass # time.sleep(0.01) removed, already sleeps in env.step

                if done:
                    break

            # --- End of Episode ---
            episode_rewards.append(episode_reward)
            avg_loss = np.mean(step_losses) if step_losses else 0
            episode_losses.append(avg_loss)

            # Decay epsilon
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

            print(f"Episode: {episode+1}/{NUM_EPISODES}, Steps: {step+1}, Reward: {episode_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.3f}")

            # Save model periodically
            if (episode + 1) % 50 == 0:
                agent.save_model(MODEL_SAVE_PATH)

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        env.close()
        print("Training finished.")
        agent.save_model(MODEL_SAVE_PATH) # Save final model

        # --- Plotting ---
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        # Add a moving average
        if len(episode_rewards) >= 10:
            moving_avg = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
            plt.plot(np.arange(9, len(episode_rewards)), moving_avg, label='10-episode MA')
            plt.legend()


        plt.subplot(1, 2, 2)
        plt.plot(episode_losses)
        plt.title('Average Training Loss per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Average Loss')
        if len(episode_losses) >= 10:
             moving_avg_loss = np.convolve(episode_losses, np.ones(10)/10, mode='valid')
             plt.plot(np.arange(9, len(episode_losses)), moving_avg_loss, label='10-episode MA')
             plt.legend()


        plt.tight_layout()
        plt.savefig("training_plots.png")
        plt.show()
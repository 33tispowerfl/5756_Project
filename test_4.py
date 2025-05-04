import os
import random
import numpy as np
import time
import math
import argparse
import matplotlib.pyplot as plt
from collections import deque

import pybullet as p
import pybullet_data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- CUDA Setup ---
# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# --- End CUDA Setup ---

# Hyperparameters
EPISODES = 300
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 10000
BATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
RENDER = False # Default render state, can be overridden by args

# Setup environment paths (adjust these to your file paths)
# It's generally better to use absolute paths or relative paths from the script location
script_dir = os.path.dirname(__file__) # Get the directory where the script is located
CAR_URDF_PATH = os.path.join(script_dir, "asset/mentorpi_description/urdf/ack.xacro")
# Using an absolute path as provided in the original script for the maze
# Ensure this path is correct on your system
MAZE_URDF_PATH = "/home/beta_frame/Documents/5756_project/asset/maze/maze_generator_ros/maze/model.urdf"
# Check if URDF files exist
if not os.path.exists(CAR_URDF_PATH):
    print(f"Warning: Car URDF not found at {CAR_URDF_PATH}")
if not os.path.exists(MAZE_URDF_PATH):
    print(f"Warning: Maze URDF not found at {MAZE_URDF_PATH}")


# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        # Ensure input tensor is on the same device as the model
        # Although usually handled before calling forward, this is a safeguard
        x = x.to(next(self.parameters()).device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store data as numpy arrays or basic types to save memory if needed
        # Conversion to tensors happens during sampling
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))

        # Convert to tensors and move to the appropriate device
        state_tensor = torch.FloatTensor(np.array(state)).to(device)
        action_tensor = torch.LongTensor(action).to(device)
        reward_tensor = torch.FloatTensor(reward).to(device)
        next_state_tensor = torch.FloatTensor(np.array(next_state)).to(device)
        done_tensor = torch.FloatTensor(done).to(device)

        return (
            state_tensor,
            action_tensor,
            reward_tensor,
            next_state_tensor,
            done_tensor
        )

    def __len__(self):
        return len(self.buffer)

# Car Environment
class CarEnv:
    def __init__(self, car_urdf_path, maze_urdf_path, render=False):
        # PyBullet setup
        self.render_mode = render
        if self.render_mode:
            self.client = p.connect(p.GUI)
            # Improve rendering (optional)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,1)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load ground plane and maze
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        try:
            self.maze_id = p.loadURDF(maze_urdf_path, basePosition=[0, 0, 0], physicsClientId=self.client)
        except p.error as e:
            print(f"Error loading maze URDF: {e}")
            print(f"Attempted path: {maze_urdf_path}")
            p.disconnect(self.client)
            raise

        # Car properties
        self.car_urdf_path = car_urdf_path
        self.car_id = None
        # Assuming indices 0,1 are front wheels (steering?) and 2,3 are rear wheels (drive?)
        # Check your URDF/XACRO to confirm joint names/indices
        # If ack.xacro has named joints, using p.getJointInfo is more robust
        # Example: Find indices by name (run once or cache)
        # num_joints = p.getNumJoints(self.car_id)
        # joint_names = [p.getJointInfo(self.car_id, i)[1].decode('utf-8') for i in range(num_joints)]
        # print(f"Car Joints: {joint_names}") # Inspect joint names
        # self.wheel_indices = [index for index, name in enumerate(joint_names) if 'wheel_joint' in name] # Adjust based on actual names
        # Using fixed indices for now as per original code
        self.wheel_indices = [1, 2, 3, 4] # Indices 1,2,3,4 might correspond to wheels in ack.xacro. VERIFY THIS.
        self.drive_wheel_indices = [3, 4] # Assuming rear wheels drive - ADJUST AS NEEDED
        self.steer_joint_indices = [1, 2] # Assuming front joints steer - ADJUST AS NEEDED (If it's Ackermann steering, control might be different)


        self.start_pos = [-1.5, -1.5, 0.07]  # Starting position
        self.start_orn = p.getQuaternionFromEuler([0, 0, 0]) # Start pointing along positive X
        try:
            self.car_id = p.loadURDF(self.car_urdf_path, self.start_pos, self.start_orn, physicsClientId=self.client)
            # Improve dynamics (optional)
            for wheel_index in self.wheel_indices:
                 p.changeDynamics(self.car_id, wheel_index, lateralFriction=1.0) # Increase tire grip
        except p.error as e:
            print(f"Error loading car URDF: {e}")
            print(f"Attempted path: {self.car_urdf_path}")
            p.disconnect(self.client)
            raise


        # Goal position (adjust based on your maze)
        self.goal_position = [1.5, 1.5, 0.0]  # Example goal position

        # Define action space (discrete actions)
        # 0: Forward, 1: Left, 2: Right
        # Removed backward and stop for simplicity, add back if needed
        self.action_size = 3
        # State: [car_x, car_y, car_z, qx, qy, qz, qw, rel_goal_x, rel_goal_y, rel_goal_z]
        self.state_size = 10

        # Physics parameters
        self.max_force = 20 # Max force applied to wheels
        self.max_velocity = 20 # Target velocity for driving
        self.steer_angle = 0.5 # Max steering angle in radians

        # Reset environment
        self.last_distance_to_goal = float('inf')
        self.reset()

        # Visualization of goal
        self.goal_id = None
        if self.render_mode:
            self.goal_visual = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 0.7], physicsClientId=self.client
            )
            self.goal_id = p.createMultiBody(
                baseVisualShapeIndex=self.goal_visual,
                basePosition=self.goal_position,
                baseCollisionShapeIndex=-1,
                 physicsClientId=self.client
            )

    def reset(self):
        # Reset robot pose
        p.resetBasePositionAndOrientation(self.car_id, self.start_pos, self.start_orn, physicsClientId=self.client)
        # Reset robot velocities
        p.resetBaseVelocity(self.car_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0], physicsClientId=self.client)
        # Ensure joints are reset too (including steering)
        for joint_index in range(p.getNumJoints(self.car_id, physicsClientId=self.client)):
             p.resetJointState(self.car_id, joint_index, targetValue=0, targetVelocity=0, physicsClientId=self.client)

        # Needed to get first state correctly after reset
        p.stepSimulation(physicsClientId=self.client)
        state = self._get_state()
        self.last_distance_to_goal = self._get_distance_to_goal(state)
        return state

    def _get_state(self):
        car_pos, car_orn_quat = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.client)

        # Calculate relative goal position
        goal_relative = [
            self.goal_position[0] - car_pos[0],
            self.goal_position[1] - car_pos[1],
            self.goal_position[2] - car_pos[2] # Keep Z for now, might be useful if terrain changes
        ]

        # Combine position, orientation quaternion and goal info
        state = list(car_pos) + list(car_orn_quat) + goal_relative
        return np.array(state, dtype=np.float32) # Use float32 for consistency with PyTorch

    def _get_distance_to_goal(self, state):
         # Calculate from state to avoid extra pybullet call if state is already computed
        car_pos = state[0:3]
        distance_to_goal = math.sqrt(
            (car_pos[0] - self.goal_position[0])**2 +
            (car_pos[1] - self.goal_position[1])**2
            # Optionally include Z: + (car_pos[2] - self.goal_position[2])**2
        )
        return distance_to_goal

    def step(self, action):
        # Apply action using Ackermann steering logic (adjust if your URDF uses a different mechanism)
        target_velocity = 0
        steering_target = 0

        if action == 0:  # Forward
            target_velocity = self.max_velocity
            steering_target = 0
        elif action == 1:  # Left turn (while moving forward)
            target_velocity = self.max_velocity * 0.8 # Slow down slightly when turning
            steering_target = self.steer_angle
        elif action == 2:  # Right turn (while moving forward)
            target_velocity = self.max_velocity * 0.8 # Slow down slightly when turning
            steering_target = -self.steer_angle
        # Add Action 3: Backward if needed
        # elif action == 3: # Backward
        #    target_velocity = -self.max_velocity
        #    steering_target = 0
        # Add Action 4: Stop if needed
        # elif action == 4: # Stop
        #    target_velocity = 0
        #    steering_target = 0 # Keep wheels straight

        # Set steering joint angles (assuming indices 1, 2 are front steering joints)
        # This depends heavily on your URDF's joint setup
        for joint_index in self.steer_joint_indices:
            p.setJointMotorControl2(self.car_id, joint_index, p.POSITION_CONTROL,
                                   targetPosition=steering_target, force=self.max_force,
                                   physicsClientId=self.client)

        # Set drive wheel velocities (assuming indices 3, 4 are rear drive wheels)
        for wheel_index in self.drive_wheel_indices:
            p.setJointMotorControl2(self.car_id, wheel_index, p.VELOCITY_CONTROL,
                                   targetVelocity=target_velocity, force=self.max_force,
                                   physicsClientId=self.client)

        # Step simulation
        # A single step might be enough if dynamics are stable, but multiple can help
        num_sim_steps = 10
        for _ in range(num_sim_steps):
            p.stepSimulation(physicsClientId=self.client)
            if self.render_mode:
                time.sleep(1/240.) # Standard PyBullet sleep time

        # Get new state
        next_state = self._get_state()
        car_pos = next_state[0:3] # Extract from state
        car_orn_quat = next_state[3:7] # Extract from state

        # Calculate reward
        distance_to_goal = self._get_distance_to_goal(next_state)

        # Check for collision with maze
        collision = False
        # Check contacts specifically with the maze body
        contact_points = p.getContactPoints(bodyA=self.car_id, bodyB=self.maze_id, physicsClientId=self.client)
        if len(contact_points) > 0:
            collision = True
            #print("Collision detected!")

        # Check if car is flipped (using orientation)
        # Check roll (around X) and pitch (around Y) angles
        euler = p.getEulerFromQuaternion(car_orn_quat)
        # Increased tolerance slightly, adjust as needed
        flipped = abs(euler[0]) > math.radians(60) or abs(euler[1]) > math.radians(60)
        if flipped:
            #print("Car flipped!")
            pass


        # --- Reward Shaping ---
        reward = 0
        done = False

        # 1. Reward for getting closer to the goal (potential-based shaping)
        # Compare current distance to the distance in the previous step
        distance_delta = self.last_distance_to_goal - distance_to_goal
        reward += distance_delta * 5.0 # Scale factor for progress reward

        # 2. Penalty for distance (sparse penalty, encourages reaching the goal faster)
        # reward -= distance_to_goal * 0.01 # Small penalty for existing distance

        # 3. Penalty for collision
        collision_penalty = -50.0
        if collision:
            reward += collision_penalty
            done = True

        # 4. Penalty for flipping
        flip_penalty = -20.0
        if flipped:
            reward += flip_penalty
            # Optionally end the episode if flipped:
            # done = True

        # 5. Large reward for reaching the goal
        goal_reward = 200.0
        goal_threshold = 0.5 # meters
        if distance_to_goal < goal_threshold:
            reward += goal_reward
            done = True
            print(f"Goal reached! Distance: {distance_to_goal:.3f}")

        # 6. Small penalty per step (encourages efficiency)
        time_penalty = -0.1
        reward += time_penalty

        # 7. Check if out of bounds (define reasonable bounds for your maze)
        max_coord = 5.0 # Example boundary
        oob_penalty = -50.0
        if abs(car_pos[0]) > max_coord or abs(car_pos[1]) > max_coord or car_pos[2] < -0.1: # Also check if fell through floor
            reward += oob_penalty
            done = True
            #print("Out of bounds!")

        # --- End Reward Shaping ---

        # Update last distance for next step's reward calculation
        self.last_distance_to_goal = distance_to_goal

        info = {"distance": distance_to_goal, "collision": collision, "flipped": flipped}

        return next_state, reward, done, info

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

# Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Initialize networks and move them to the selected device
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only for inference

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

        # Initialize replay buffer
        self.memory = ReplayBuffer(MEMORY_SIZE)

        # Initialize epsilon for exploration
        self.epsilon = EPSILON_START

        # Initialize step counter for target network updates
        self.steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size) # Explore

        # Exploit: select the action with the highest Q-value
        with torch.no_grad(): # No need to track gradients here
            # Convert state (numpy array) to tensor and move to device
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            # Get Q-values from the policy network
            q_values = self.policy_net(state_tensor)
            # Select the action with the maximum Q-value
            action = q_values.max(1)[1].item() # .item() gets the Python number
            return action

    def update_epsilon(self):
        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def remember(self, state, action, reward, next_state, done):
        # Add experience to replay buffer
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        # Only start learning when the buffer has enough samples
        if len(self.memory) < BATCH_SIZE:
            return None # Return None or 0 if no learning happened

        # Sample a batch from the replay buffer (tensors are already on the correct device)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(BATCH_SIZE)

        # Compute current Q values: Q(s, a)
        # Get Q values for all actions from policy_net, then select the Q value for the action taken
        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute next Q values from target net: max_a' Q_target(s', a')
        # Use torch.no_grad() because we don't need gradients for the target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]

        # Compute the expected Q values (Bellman equation)
        # target = r + gamma * max_a' Q_target(s', a') * (1 - done)
        expected_q_values = reward_batch + (1 - done_batch) * GAMMA * next_q_values

        # Compute loss (e.g., Smooth L1 loss)
        # Detach expected_q_values as they are considered fixed targets
        loss = F.smooth_l1_loss(q_values, expected_q_values.detach())

        # Optimize the model
        self.optimizer.zero_grad() # Clear old gradients
        loss.backward() # Compute gradients

        # Gradient clipping (optional but often helpful)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step() # Update network weights

        # Update target network periodically
        self.steps += 1
        if self.steps % UPDATE_TARGET_EVERY == 0:
            # Copy weights from policy_net to target_net
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item() # Return the loss value for monitoring

    def save(self, filename):
        # Save model state dictionaries and optimizer state
        # State dicts are automatically saved from the device they are on
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        # Load model state dictionaries and optimizer state
        # Use map_location to load the model onto the correct device (CPU or GPU)
        try:
            checkpoint = torch.load(filename, map_location=device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            # Ensure target network is in evaluation mode after loading
            self.target_net.eval()
            # Optionally move models to device again, though map_location should handle it
            self.policy_net.to(device)
            self.target_net.to(device)
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {filename}")
        except Exception as e:
             print(f"Error loading model: {e}")


# Training function
def train(render_mode):
    # Create environment
    env = CarEnv(CAR_URDF_PATH, MAZE_URDF_PATH, render=render_mode)

    # Create agent
    agent = DQNAgent(env.state_size, env.action_size)

    # Training metrics
    rewards_history = []
    losses_history = []
    epsilons_history = []
    distances_history = []
    episode_durations = [] # Track steps per episode

    # Ensure model saving directory exists
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)


    for episode in range(EPISODES):
        start_time = time.time()
        state = env.reset()
        total_reward = 0
        current_loss = 0
        steps_in_episode = 0
        done = False

        while not done:
            # Select action based on current state and epsilon-greedy policy
            action = agent.select_action(state)

            # Take action in the environment
            next_state, reward, done, info = env.step(action)

            # Store the transition in the replay buffer
            agent.remember(state, action, reward, next_state, done)

            # Perform one step of the optimization (on the policy network)
            loss = agent.learn()
            if loss is not None:
                 current_loss += loss # Accumulate loss over the episode steps where learning occurs

            # Move to the next state
            state = next_state
            total_reward += reward
            steps_in_episode += 1

            # Optional: add a maximum step limit per episode
            # if steps_in_episode > MAX_STEPS_PER_EPISODE:
            #    done = True

            if done:
                break

        # Update epsilon for exploration rate decay
        agent.update_epsilon()

        # Record metrics for the episode
        duration = time.time() - start_time
        episode_durations.append(duration)
        rewards_history.append(total_reward)
        # Average loss over steps where learning happened
        avg_loss = current_loss / steps_in_episode if steps_in_episode > 0 and current_loss > 0 else 0
        losses_history.append(avg_loss)
        epsilons_history.append(agent.epsilon)
        distances_history.append(info.get('distance', -1)) # Use .get for safety

        # Print progress
        print(f"Episode: {episode+1}/{EPISODES}, Steps: {steps_in_episode}, Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.3f}, Dist: {info.get('distance', -1):.2f}, Time: {duration:.2f}s")

        # Save model periodically
        if (episode + 1) % 100 == 0:
            model_save_path = os.path.join(save_dir, f"dqn_car_model_episode_{episode+1}.pth")
            agent.save(model_save_path)

    # Save final model
    final_model_path = os.path.join(save_dir, "dqn_car_model_final.pth")
    agent.save(final_model_path)

    # Close environment
    env.close()

    # Plot results
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 4, 1)
    plt.plot(rewards_history)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    # Add moving average for rewards
    if len(rewards_history) >= 10:
        moving_avg = np.convolve(rewards_history, np.ones(10)/10, mode='valid')
        plt.plot(np.arange(9, len(rewards_history)), moving_avg, label='10-ep Moving Avg')
        plt.legend()


    plt.subplot(1, 4, 2)
    plt.plot(losses_history)
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')

    plt.subplot(1, 4, 3)
    plt.plot(distances_history)
    plt.title('Final Distance to Goal per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    plt.ylim(bottom=0) # Distance cannot be negative

    plt.subplot(1, 4, 4)
    plt.plot(epsilons_history)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')


    plt.tight_layout()
    plt.savefig('training_results.png')
    print("Training results graph saved as training_results.png")
    plt.show()

# Evaluation function
def evaluate(model_path, num_episodes=10, render_mode=True):
    # Create environment with rendering enabled for evaluation
    env = CarEnv(CAR_URDF_PATH, MAZE_URDF_PATH, render=render_mode)

    # Create agent
    agent = DQNAgent(env.state_size, env.action_size)

    # Load the trained model
    agent.load(model_path)
    # Set epsilon to 0 for deterministic behavior during evaluation
    agent.epsilon = 0.0
    # Ensure the policy network is in evaluation mode
    agent.policy_net.eval()


    total_rewards = []
    final_distances = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps_in_episode = 0

        while not done:
            # Select action using the learned policy (no exploration)
            action = agent.select_action(state)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Update state and total reward
            state = next_state
            total_reward += reward
            steps_in_episode += 1

            # Optional step limit for evaluation
            # if steps_in_episode > MAX_EVAL_STEPS:
            #    done = True

            if done:
                break

        total_rewards.append(total_reward)
        final_distances.append(info.get('distance', -1))
        print(f"Evaluation Episode {episode + 1}/{num_episodes}, Steps: {steps_in_episode}, Reward: {total_reward:.2f}, Final Distance: {info.get('distance', -1):.2f}")
        # Add a small delay between episodes if rendering
        if render_mode:
            time.sleep(1)


    # Close environment
    env.close()

    # Print summary statistics
    print("\n--- Evaluation Summary ---")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Average Final Distance: {np.mean(final_distances):.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")
    print("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN Car Navigation in PyBullet')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--eval', action='store_true', help='Evaluate the trained agent')
    parser.add_argument('--model', type=str, default='saved_models/dqn_car_model_final.pth', help='Path to the model file for evaluation')
    parser.add_argument('--render', action='store_true', help='Enable rendering during training or evaluation')
    parser.add_argument('--episodes', type=int, default=EPISODES, help=f'Number of training episodes (default: {EPISODES})')

    args = parser.parse_args()

    # Set global render flag based on argument
    RENDER_ARG = False # Store render arg

    # Update EPISODES if provided
    EPISODES = args.episodes


    if True:
        print("Starting training...")
        # Pass render flag to train function
        train(render_mode=RENDER_ARG)
    elif args.eval:
        print(f"Starting evaluation using model: {args.model}")
        # Pass render flag (usually True for eval) and model path
        evaluate(args.model, num_episodes=10, render_mode=RENDER_ARG)
    else:
        print("Please specify --train or --eval mode.")
        parser.print_help()
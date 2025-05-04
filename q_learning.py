import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import math
import random
from collections import defaultdict, deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Configuration ---

# File Paths (Ensure URDF is converted from XACRO)
CAR_URDF_PATH = "./asset/mentorpi_description/urdf/ack.xacro"
MAZE_URDF_PATH = "/home/beta_frame/Documents/5756_project/asset/maze/maze_generator_ros/maze/model.urdf"

# Driving Joints (Identify Left vs Right)
LEFT_WHEEL_JOINT_NAMES = ["wheel_lf_Joint", "wheel_lb_Joint"]
RIGHT_WHEEL_JOINT_NAMES = ["wheel_rf_Joint", "wheel_rb_Joint"]

# Environment Parameters
GOAL_POS = np.array([1.5, 1.5])
START_POS = np.array([-1.5, -1.5, 0.1])
GOAL_THRESHOLD = 0.4
STATE_SIZE = 3  # Input state for DQN: [x, y, yaw]

# Car Control Parameters
TARGET_VELOCITY = 5.0
TURN_VELOCITY_FACTOR = 0.3  # Speed of inner wheels during turn (Factor of TARGET_VELOCITY)
MAX_FORCE = 50

# --- Action Space Definition ---
# Define action space explicitly with named constants
FORWARD = 0
LEFT = 1
RIGHT = 2
STOP = 3
ACTION_SPACE_SIZE = 4

# --- DQN Parameters ---
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
LR = 5e-4
TAU = 1e-3
UPDATE_EVERY = 4
TARGET_UPDATE_EVERY = 100

# Epsilon-Greedy Parameters
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

# Training Parameters
NUM_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 300

# Rewards
GOAL_REWARD = 100
COLLISION_REWARD = -100
STEP_REWARD = -0.1

# Device Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Helper Functions ---

def get_joint_index_by_name(body_id, name):
    """Finds the index of a joint by its name."""
    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        if info[1].decode('utf-8') == name:
            return info[0]
    print(f"Warning: Joint '{name}' not found!")
    return -1

def get_yaw_from_orientation(orientation_quat):
    """Extracts the yaw angle from a quaternion."""
    euler = p.getEulerFromQuaternion(orientation_quat)
    return euler[2]  # Yaw is typically the third element (around Z-axis)

# --- DQN Network ---
class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- Replay Buffer ---
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        np.random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        if len(self.memory) < self.batch_size:
            return None

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# --- DQN Agent ---
class DQNAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.target_update_counter = 0
        self.epsilon = EPSILON_START

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                if experiences is not None:
                    self.learn(experiences, GAMMA)

        # Update target network
        self.target_update_counter += 1
        if self.target_update_counter >= TARGET_UPDATE_EVERY:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
            self.target_update_counter = 0

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def update_epsilon(self):
        """Decay epsilon for epsilon-greedy policy."""
        self.epsilon = max(EPSILON_END, EPSILON_DECAY * self.epsilon)

# --- Environment ---
class CarEnvironment:
    def __init__(self, gui=True):
        if gui:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # load ground + maze
        self.plane_id, self.maze_id = self.load_maze()
        self.car_id = None
        self.joint_indices = {}
        _ = self.reset()

    def load_maze(self):
        plane_id = p.loadURDF("plane.urdf")
        try:
            maze_id = p.loadURDF(
                MAZE_URDF_PATH,
                basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True
            )
        except p.error as e:
            print(f"Error loading maze URDF: {e}")
            maze_id = None
        return plane_id, maze_id

    def reset(self):
        if self.car_id is not None:
            p.removeBody(self.car_id)
        # spawn higher so it doesn’t pierce the floor
        spawn_z = 0.5
        self.car_id = p.loadURDF(
            CAR_URDF_PATH,
            [START_POS[0], START_POS[1], spawn_z],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        # settle
        for _ in range(50):
            p.stepSimulation()

        pos, orient = p.getBasePositionAndOrientation(self.car_id)
        yaw = get_yaw_from_orientation(orient)
        return np.array([pos[0], pos[1], yaw])

    def step(self, action):
        reward = self._apply_action(action)
        p.stepSimulation()  # you usually only need one sim-tick per step
        pos, orient = p.getBasePositionAndOrientation(self.car_id)
        yaw = get_yaw_from_orientation(orient)
        next_state = np.array([pos[0], pos[1], yaw])

        # goal check…
        done = False
        if np.linalg.norm(next_state[:2] - GOAL_POS) < GOAL_THRESHOLD:
            reward += GOAL_REWARD
            done = True

        # now *only* walls
        if self.maze_id is not None:
            contacts = p.getContactPoints(bodyA=self.car_id, bodyB=self.maze_id)
            if contacts:
                reward += COLLISION_REWARD
                done = True

        return next_state, reward, done


    def _apply_action(self, action):
        """Apply action to control the car wheels.

        Actions:
        0 (FORWARD): Move forward (all wheels at TARGET_VELOCITY)
        1 (LEFT): Turn left (right wheels faster than left)
        2 (RIGHT): Turn right (left wheels faster than right)
        3 (STOP): Stop all wheels

        Returns:
        float: Step reward
        """
        left_velocity = 0
        right_velocity = 0

        if action == FORWARD:
            left_velocity = TARGET_VELOCITY
            right_velocity = TARGET_VELOCITY
        elif action == LEFT:
            left_velocity = TARGET_VELOCITY * TURN_VELOCITY_FACTOR
            right_velocity = TARGET_VELOCITY
        elif action == RIGHT:
            left_velocity = TARGET_VELOCITY
            right_velocity = TARGET_VELOCITY * TURN_VELOCITY_FACTOR
        elif action == STOP:
            left_velocity = 0
            right_velocity = 0

        # Apply velocities to wheels
        for joint_name in LEFT_WHEEL_JOINT_NAMES:
            if joint_name in self.joint_indices:
                p.setJointMotorControl2(self.car_id, self.joint_indices[joint_name],
                                       p.VELOCITY_CONTROL, targetVelocity=left_velocity, force=MAX_FORCE)

        for joint_name in RIGHT_WHEEL_JOINT_NAMES:
            if joint_name in self.joint_indices:
                p.setJointMotorControl2(self.car_id, self.joint_indices[joint_name],
                                       p.VELOCITY_CONTROL, targetVelocity=right_velocity, force=MAX_FORCE)

        # Return step penalty
        return STEP_REWARD

# --- Training Function ---
def train_dqn():
    # Create environment and agent
    env = CarEnvironment(gui=True)
    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SPACE_SIZE, seed=0)

    # Training statistics
    scores = []
    scores_window = deque(maxlen=100)

    for episode in range(1, NUM_EPISODES+1):
        state = env.reset()
        score = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            # Select and execute action
            action = agent.act(state, agent.epsilon)
            next_state, reward, done = env.step(action)

            # Update agent
            agent.step(state, action, reward, next_state, done)

            # Update state and score
            state = next_state
            score += reward

            if done:
                break

        # Update epsilon
        agent.update_epsilon()

        # Save score
        scores.append(score)
        scores_window.append(score)

        # Print progress
        if episode % 100 == 0:
            print(f'Episode {episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {agent.epsilon:.2f}')

    return agent, scores

# --- Testing Function ---
def test_agent(agent, num_episodes=10):
    env = CarEnvironment(gui=True)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Get action (no exploration)
            action = agent.act(state, eps=0.0)

            # Take action
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Update state
            state = next_state

            # Slow down visualization
            time.sleep(0.05)

        print(f'Episode {episode+1}: Total Reward = {total_reward}')

# --- Main Execution ---
if __name__ == "__main__":
    # train headless
    print("Starting training…")
    agent, scores = train_dqn()

    # save
    torch.save(agent.qnetwork_local.state_dict(), "dqn_car_navigation.pth")

    # then *test* in GUI
    print("Testing agent…")
    test_agent(agent, gui=True)

    p.disconnect()
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
RENDER = False

# Setup environment paths (adjust these to your file paths)
CAR_URDF_PATH = "./asset/mentorpi_description/urdf/ack.xacro"
MAZE_URDF_PATH = "/home/beta_frame/Documents/5756_project/asset/maze/maze_generator_ros/maze/model.urdf"

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
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
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load ground plane and maze
        self.plane_id = p.loadURDF("plane.urdf")
        self.maze_id = p.loadURDF(maze_urdf_path, basePosition=[0, 0, 0])



        # Car properties
        self.car_urdf_path = car_urdf_path
        self.car_id = None
        self.wheel_indices = [1,2,3,4]

        self.start_pos = [-1.5, -1.5, 0.07]  # Starting position
        self.start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.car_id = p.loadURDF(self.car_urdf_path, basePosition=self.start_pos, baseOrientation=self.start_orn)

        # Goal position (adjust based on your maze)
        self.goal_position = [1.5, 1.5, 0.0]  # Example goal position

        # Define action space (discrete actions)
        # 0: Forward, 1: Backward, 2: Left, 3: Right, 4: Stop
        self.action_size = 3
        self.state_size = 10  # position (x, y, z) + orientation (x, y, z, w) + goal relative(3)

        # Reset environment
        self.reset()

        # Visualization of goal
        if self.render_mode:
            self.goal_visual = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 0.7]
            )
            self.goal_id = p.createMultiBody(
                baseVisualShapeIndex=self.goal_visual,
                basePosition=self.goal_position,
                baseCollisionShapeIndex=-1
            )

    def reset(self):
        # Reset robot pose
        p.resetBasePositionAndOrientation(self.car_id, self.start_pos, self.start_orn, physicsClientId=self.client)
        # Reset robot velocities
        p.resetBaseVelocity(self.car_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0], physicsClientId=self.client)
        # Ensure joints are reset too
        for joint_index in self.wheel_indices:
             p.resetJointState(self.car_id, joint_index, targetValue=0, targetVelocity=0, physicsClientId=self.client)

        # Needed to get first state correctly after reset
        p.stepSimulation(physicsClientId=self.client)

        return self._get_state()

    def _get_state(self):
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id)

        # Add goal relative position for better learning
        goal_relative = [
            self.goal_position[0] - car_pos[0],
            self.goal_position[1] - car_pos[1],
            self.goal_position[2] - car_pos[2]
        ]

        # Combine position, orientation and goal info
        state = list(car_pos) + list(car_orn) + goal_relative
        return np.array(state)

    def step(self, action):
        # Apply action
        vel = 10.0  # Max velocity

        # Reset velocities
        for wheel in self.wheel_indices:
            p.setJointMotorControl2(self.car_id, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        # Apply actions
        if action == 0:  # Forward
            for wheel in self.wheel_indices[2:]:
                p.setJointMotorControl2(self.car_id, wheel, p.VELOCITY_CONTROL, targetVelocity=vel, force=10)
        elif action == 1:  # Left turn
            # Assuming 4 wheels with indices 0,1 = left side, 2,3 = right side
            p.setJointMotorControl2(self.car_id, self.wheel_indices[2], p.VELOCITY_CONTROL, targetVelocity=-vel*5, force=10)
            p.setJointMotorControl2(self.car_id, self.wheel_indices[3], p.VELOCITY_CONTROL, targetVelocity=vel*15, force=10)
        elif action == 2:  # Right turn
            p.setJointMotorControl2(self.car_id, self.wheel_indices[2], p.VELOCITY_CONTROL, targetVelocity=vel*15, force=10)
            p.setJointMotorControl2(self.car_id, self.wheel_indices[3], p.VELOCITY_CONTROL, targetVelocity=-vel*5, force=10)
        # Action 4 is stop (do nothing)

        # Step simulation
        for _ in range(10):  # Multiple steps for stability
            p.stepSimulation()
            if self.render_mode:
                time.sleep(1/240.)  # Slow down rendering

        # Get new state
        next_state = self._get_state()

        # Calculate reward
        car_pos = p.getBasePositionAndOrientation(self.car_id)[0]

        # Distance to goal
        distance_to_goal = math.sqrt(
            (car_pos[0] - self.goal_position[0])**2 +
            (car_pos[1] - self.goal_position[1])**2
        )

        # Check for collision with maze
        collision = False
        contact_points = p.getContactPoints(self.car_id, self.maze_id)
        if len(contact_points) > 0:
            collision = True

        # Check if car is flipped
        _, car_orn = p.getBasePositionAndOrientation(self.car_id)
        euler = p.getEulerFromQuaternion(car_orn)
        flipped = abs(euler[0]) > 0.5 or abs(euler[1]) > 0.5

        # Define reward
        reward = 0
        done = False

        # Reward for getting closer to goal
        reward -= distance_to_goal * 0.1

        # Penalty for collision
        if collision:
            reward -= 25
            done = True

        # Penalty for flipping
        if flipped:
            reward -= 5
            done = True

        # Check if goal reached
        if distance_to_goal < 0.5:
            reward += 100
            done = True
            print("Goal reached!")

        # Penalty for time
        reward -= 0.01

        # Check if out of bounds
        if abs(car_pos[0]) > 10 or abs(car_pos[1]) > 10:
            reward -= 10
            done = True

        info = {"distance": distance_to_goal}

        return next_state, reward, done, info

    def close(self):
        p.disconnect(self.client)

# Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Initialize networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

        # Initialize replay buffer
        self.memory = ReplayBuffer(MEMORY_SIZE)

        # Initialize epsilon for exploration
        self.epsilon = EPSILON_START

        # Initialize step counter
        self.steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # Sample batch from replay buffer
        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)

        # Compute current Q values
        q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # Compute next Q values
        next_q_values = self.target_net(next_state).max(1)[0]
        expected_q_values = reward + (1 - done) * GAMMA * next_q_values

        # Compute loss
        loss = F.smooth_l1_loss(q_values, expected_q_values.detach())

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % UPDATE_TARGET_EVERY == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, filename):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.target_net.eval()

        self.policy_net.to(device)
        self.target_net.to(device)

# Training function
def train():
    # Create environment
    env = CarEnv(CAR_URDF_PATH, MAZE_URDF_PATH, render=RENDER)

    # Create agent
    agent = DQNAgent(env.state_size, env.action_size)

    # Training metrics
    rewards = []
    losses = []
    epsilons = []
    distances = []

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        loss = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.remember(state, action, reward, next_state, done)

            # Learn
            loss = agent.learn() or loss

            # Update state
            state = next_state
            total_reward += reward

            if done:
                break

        # Update epsilon
        agent.update_epsilon()

        # Record metrics
        rewards.append(total_reward)
        losses.append(loss)
        epsilons.append(agent.epsilon)
        distances.append(info['distance'])

        # Print progress
        if episode % 1 == 0:
            print(f"Episode: {episode}, Reward: {total_reward:.2f}, Loss: {loss:.4f}, Epsilon: {agent.epsilon:.2f}, Distance to goal: {info['distance']:.2f}")

            # Save model
            if episode % 100 == 0:
                agent.save(f"dqn_car_model_episode_{episode}.pth")

    # Save final model
    agent.save("dqn_car_model_final.pth")

    # Close environment
    env.close()

    # Plot results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title('Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 3, 2)
    plt.plot(losses)
    plt.title('Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.subplot(1, 3, 3)
    plt.plot(distances)
    plt.title('Distance to Goal')
    plt.xlabel('Episode')
    plt.ylabel('Distance')

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

# Evaluation function
def evaluate(model_path, num_episodes=10):
    # Create environment
    env = CarEnv(CAR_URDF_PATH, MAZE_URDF_PATH, render=True)

    # Create agent
    agent = DQNAgent(env.state_size, env.action_size)
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration during evaluation

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Update state
            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Evaluation Episode {episode + 1}/{num_episodes}, Reward: {total_reward:.2f}, Distance to goal: {info['distance']:.2f}")

    # Close environment
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN Car Navigation')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--eval', action='store_true', help='Evaluate the agent')
    parser.add_argument('--model', type=str, default='dqn_car_model_final.pth', help='Model path for evaluation')
    parser.add_argument('--render', action='store_true', help='Render the environment')

    args = parser.parse_args()

    # Set render mode
    RENDER = False

    if True:
        train()
    elif args.eval:
        evaluate(args.model)
    else:
        print("Please specify --train or --eval")
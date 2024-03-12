import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch.nn.functional as F

'''This code trains a simple DQN agent in the HighwayEnv environment'''

# initialize the environment and set our variables
env = gym.make('highway-v0', render_mode='rgb_array')
env.reset()
#reset the environment
obs, info = env.reset()
done = truncated = False

# print discrete action space
print(f"Action space: {env.action_space}")

# print observation space
print(obs)

# set our size variables
obs_size = np.prod(env.observation_space.shape)
action_size = env.action_space.n

# close our environment
env.close()

# check if a hardware accelerator is available, otherwise train on the cpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

'''
First Linear layer: Takes inputs of size 'obs_size' (observation space size), outputs 128 features. Acts as the input layer that begins the process of learning representations.
First ReLU activation: Applies the Rectified Linear Unit function element-wise, introducing non-linearity and allowing for complex patterns to be learned.
Second Linear layer: Takes 128 features from the previous layer as input, outputs another 128 features. It serves to further process and refine the representations.
Second ReLU activation: Another application of ReLU to introduce non-linearity, enabling the network to capture more complex relationships.
Final Linear layer: Takes 128 features from the previous processing, outputs a vector of size 'action_size', representing the estimated value of each action given the current state.
'''
class DQN(nn.Module):
    def __init__(self, obs_size, action_size):
        super(DQN, self).__init__()
        # Define the network layers
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

'''
The ReplayBuffer class is designed to store and manage the experiences (state, action, reward, next_state, done) 
collected by the agent during training. This experience replay mechanism allows the agent to learn from past experiences 
by randomly sampling a batch of experiences from the buffer. This approach helps in stabilizing the training process by 
breaking the correlation between consecutive samples and providing a diverse set of experiences for the agent to learn from. 
The buffer has a fixed capacity, and once full, older experiences are removed to make room for new ones. 
The class provides methods to add experiences to the buffer, sample a batch of experiences for training, and get the current size of the buffer.
'''
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done
    
    def __len__(self):
        return len(self.buffer)


# pass our network to the device and visualize
# model = DQN(obs_size, action_size).to(device)
# print(model)

# Instantiate the DQN and the replay buffer
dqn = DQN(obs_size, action_size).to(device)
buffer = ReplayBuffer(10000)
learning_rate = 1e-4  # Learning rate
optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)  # Use the learning rate

# Training hyperparameters
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)
batch_size = 32
episodes = 1000
gamma = 0.99
target_update = 10

# Initialize the environment, DQN model, target model, optimizer, and replay buffer
env = gym.make('highway-v0', render_mode='rgb_array')
policy_net = DQN(obs_size, action_size).to(device)
target_net = DQN(obs_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Set the target network to evaluation mode
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(10000)

def select_action(state, policy_net, epsilon):
    """
    Selects an action using an epsilon-greedy policy.

    Args:
    - state: the current state of the environment
    - policy_net: the current policy network
    - epsilon: the probability of choosing a random action, in the range [0, 1]

    Returns:
    - action: the chosen action
    """
    if random.random() > epsilon:
        # Exploitation: choose the best action
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state)
            action = q_values.max(1)[1].item()
    else:
        # Exploration: choose a random action
        action = action = env.action_space.sample()
    return action

def compute_td_loss(batch, policy_net, target_net, gamma):
    """
    Computes the TD loss for a batch of experiences.

    Args:
    - batch: a batch of experiences
    - policy_net: the current policy network
    - target_net: the target network
    - gamma: the discount factor for future rewards

    Returns:
    - loss: the computed TD loss
    """
    states, actions, rewards, next_states, dones = batch

    states = torch.FloatTensor(states).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.tensor(dones, dtype=torch.bool).to(device)

    # Current Q values
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Next Q values
    next_q_values = target_net(next_states).max(1)[0]
    next_q_values[dones] = 0.0
    next_q_values = next_q_values.detach()

    # Compute the target Q values
    target_q_values = rewards + gamma * next_q_values
    
    # TD loss
    loss = F.mse_loss(q_values, target_q_values)

    return loss

# Training loop
for episode in range(episodes):
    obs = env.reset()[0]
    obs = np.array(obs).flatten()  # Flatten the observation
    total_reward = 0
    steps = 0
    done = False
    while not done:
        steps += 1
        epsilon = epsilon_by_frame(steps)
        action = select_action(obs, policy_net, epsilon)
        next_obs, reward, done, truncated, info = env.step(action)
        next_obs = np.array(next_obs).flatten()  # Flatten the next observation
        replay_buffer.push(obs, action, reward, next_obs, done)
        
        obs = next_obs
        total_reward += reward
        
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            loss = compute_td_loss(batch, policy_net, target_net, gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if steps % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode: {episode}, Total Reward: {total_reward}")

    # Save the model periodically and at the end of training
    if episode % 100 == 0 or episode == episodes - 1:
        torch.save(policy_net.state_dict(), f"policy_net_episode_{episode}.pth")

# Close the environment after training
env.close()
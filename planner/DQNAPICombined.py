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
from openai import OpenAI
import gymnasium as gym
import matplotlib.pyplot as plt
import warnings

# Ignore deprecation warnings
warnings.filterwarnings('ignore')

# OpenAI ChatCompletions API client initialization
client = OpenAI(
    api_key=""
)

# Select action function, calls the ChatCompletions endpoint

def selectAction(prompt):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "Your goal is to drive safely and efficiently. You are directing the ego vehicle in our simulation, selecting actions when prompted. Respond with the action name only, and nothing else."},
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message.content.strip()

# Helper functions to check if we need to select actions other than "IDLE"

def is_too_close(obs, threshold=20):  # Adjusted threshold to 20
    """Check if any other vehicle is within a certain distance in meters."""
    for i in range(1, len(obs)):
        distance = abs(obs[0, 1] - obs[i, 1]) * 100  # Assuming the distance needs to be calculated from the ego vehicle
        if distance < threshold:
            return True
    return False

def is_off_center(y_pos, lane_width=4.0):
    """Check if the ego vehicle is significantly off-center in its lane."""
    lane_center = (round(y_pos / lane_width) * lane_width) + (lane_width / 2)
    off_center_threshold = lane_width * 0.25  # 25% of the lane width as tolerance
    return abs(y_pos - lane_center) > off_center_threshold

def is_speed_below_target(x_vel, target_speed=20):  # Adjusted function name for clarity
    """Check if the ego vehicle's speed is below the target speed."""
    return x_vel < target_speed  # Adjust as necessary

# HighwayEnv config & setup
# Our config for the simulator

config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "order": "sorted"
    }
}

env = gym.make('highway-v0', render_mode='rgb_array')
env.configure(config)

# Reset the environment

obs, info = env.reset()
done = truncated = False

# Discrete action space

ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

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
# episodes = 1000
episodes = 5 # for API calls and simple model, we test with 5 episodes
gamma = 0.99
target_update = 10

# Initialize the environment, DQN model, target model, optimizer, and replay buffer
env = gym.make('highway-v0', render_mode='rgb_array')
env.configure(config)
policy_net = DQN(obs_size, action_size).to(device)
target_net = DQN(obs_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Set the target network to evaluation mode
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(10000)

def select_action(obs, y_pos_unnormalized, x_vel_unnormalized, state, policy_net, epsilon):
    """
    Selects an action using an epsilon-greedy policy.

    Args:
    - state: the current state of the environment
    - policy_net: the current policy network
    - epsilon: the probability of choosing a random action, in the range [0, 1]

    Returns:
    - action: the chosen action
    """
    # Logic to check if we need to select an action other than "IDLE"

    # Check conditions
    too_close = is_too_close(obs)
    off_center = is_off_center(y_pos_unnormalized)
    speed_below_target = is_speed_below_target(x_vel_unnormalized)

    # Decision logic
    
    if too_close or off_center or speed_below_target:
        action = env.action_type.actions_indexes[selectAction(prompt)]  # Call API for action decision
    else:
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state)
                action = q_values.max(1)[1].item()
        else:
            action = env.action_space.sample()
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
    original_obs = env.reset()[0]  # Store the full output of env.reset()
    obs = np.array(original_obs[0]).flatten()  # Extract and flatten the first element for the RL model
    total_reward = 0
    steps = 0
    done = False
    while not done:
        steps += 1
        epsilon = epsilon_by_frame(steps)
        
        prompt = ""

        availableActions = env.get_available_actions()
        # Translate action numbers to names using ACTIONS_ALL
        availableActionNames = [ACTIONS_ALL[action] for action in availableActions]
        prompt += f"Ego vehicle's available Actions: {availableActionNames}\n"

        # Use original_obs[0] for prompt generation to maintain access to the full original observation
        x_vel = original_obs[0][3]  # Adjusted to access the first element
        x_vel_unnormalized = x_vel * 20
        y_vel = original_obs[0][4]
        y_vel_unnormalized = y_vel * 20

        x_pos = original_obs[0][1]
        x_pos_unnormalized = x_pos * 100
        y_pos = original_obs[0][2]
        y_pos_unnormalized = y_pos * 100

        lane_id = round(y_pos_unnormalized / 4.0) + 1

        prompt += f"Ego vehicle: X Position: {x_pos_unnormalized:.2f}, Y Position: {y_pos_unnormalized:.2f}, X Velocity: {x_vel_unnormalized:.2f}, Y Velocity: {y_vel_unnormalized:.2f}, Current Lane: {lane_id}\n"

        # Iterate through other vehicles in the original_obs[0] part of the observation space
        for i in range(1, len(original_obs)):
            x_position = x_pos_unnormalized + (original_obs[i][1] * 100)
            y_position = y_pos_unnormalized + (original_obs[i][2] * 100)
            x_velocity = x_vel_unnormalized + (original_obs[i][3] * 20)
            y_velocity = y_vel_unnormalized + (original_obs[i][4] * 20)
            laneid = round(y_position / 4.0) + 1 #relative lanes
            prompt += f"Vehicle {i}: X Position: {x_position:.2f}, Y Position: {y_position:.2f}, X Velocity: {x_velocity:.2f}, Y Velocity: {y_velocity:.2f}, Current Lane: {laneid}\n"

        # print(prompt)  # Print the completed prompt at each step, used for troubleshooting

        # action = select_action(obs, policy_net, epsilon)  # Use the flattened obs for action selection
        action = select_action(original_obs, y_pos_unnormalized, x_vel_unnormalized, obs, policy_net, epsilon)
        next_original_obs, reward, done, truncated, info = env.step(action)
        next_obs = np.array(next_original_obs[0]).flatten()  # Flatten the first element of the next observation
        replay_buffer.push(obs, action, reward, next_obs, done)
        
        obs = next_obs
        original_obs = next_original_obs  # Update the full original_obs with the new state
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

    if episode % 100 == 0 or episode == episodes - 1:
        torch.save(policy_net.state_dict(), f"policy_net_episode_{episode}.pth")

# Close the environment after training
env.close()
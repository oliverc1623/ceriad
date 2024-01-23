import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from CustomEnvRacecar import CustomEnvRaceCar
import torch
import os
import matplotlib.pyplot as plt
from IPython import display

# For custom scenarios:
env = CustomEnvRaceCar(
    id="SingleAgentRaceEnv-v0",
    scenario="rgb-camera-scenario.yaml",
    render_mode="rgb_array_follow",
)

model = PPO.load("ppo-racecar")

obs, info = env.reset()
image = env.render()
plt.imsave(f"frames/frame_{0:04}.png", image)

done = False
total_rewards=0
t = 1
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    total_rewards += rewards
    image = env.render()
    plt.imsave(f"frames/frame_{t:04}.png", image)
    done = done
    t += 1

print(f"Total rewards: {total_rewards}")
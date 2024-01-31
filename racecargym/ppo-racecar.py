import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from CustomEnvRacecar import CustomEnvRaceCar
import torch
import os

device = "mps" if torch.backends.mps.is_available() else "cpu"

# For custom scenarios:
env = CustomEnvRaceCar(
    id="SingleAgentRaceEnv-v0",
    scenario="rgb-camera-scenario.yaml",
    render_mode="rgb_array_follow",
)

log_dir = f"tmp"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

print(check_env(env))

model = PPO("CnnPolicy", env, verbose=1, device="mps")
model.learn(total_timesteps=25000)
model.save("ppo-racecar")
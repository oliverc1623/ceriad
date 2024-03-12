import gymnasium as gym
import torch
import time
from stable_baselines3 import PPO
from CarlaEnv import CustomEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import os

model = PPO.load("carla-lanefollow-empty")
vec_env = CustomEnv(ego_vehicle='car1')
obs, info = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = vec_env.step(action)
    #vec_env.render()

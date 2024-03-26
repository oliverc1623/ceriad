import gymnasium as gym
import torch
import time
from stable_baselines3 import PPO
from CarlaEnv import CustomEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import os
import matplotlib.pyplot as plt
import numpy as np

model = PPO.load("carla-lanefollow-empty_trial0")
vec_env = CustomEnv(ego_vehicle='car1')
obs, info = vec_env.reset()
print(obs.shape)
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = vec_env.step(action)
    if dones:
        obs, info = vec_env.reset()

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
fig,ax = plt.subplots(1,1)
myobj = ax.imshow(env.render())
done = False
total_rewards=0
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    total_rewards += rewards
    image = env.render()
    myobj.set_data(image)
    fig.canvas.draw_idle()
    plt.pause(.01)

    done = done

print(f"Total rewards: {total_rewards}")
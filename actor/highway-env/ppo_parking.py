# %%
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env


# %%

# Create the environment
env = gym.make("parking-v0")

#%%
# Optional: Check if the environment follows the Gym API
check_env(env)

#%%

# Define the model
model = PPO("MultiInputPolicy", env, verbose=1)

#%%

# Callbacks
log_dir = Path("/root/../pvcvolume/highway-env-ppo-logs/")
if not log_dir.exists():
    log_dir.mkdir(parents=True)

#%%

# Stop training once the agent reaches a reward threshold
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=40, verbose=1)

eval_callback = EvalCallback(
    env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=10000,
    deterministic=True,
    render=False,
    callback_on_new_best=callback_on_best,
)

# Train the agent
model.learn(total_timesteps=int(1e6), callback=eval_callback, progress_bar=True)

env.close()

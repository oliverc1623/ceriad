import gymnasium as gym
import torch
import time
from stable_baselines3 import PPO
from CarlaEnv import CustomEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import os

def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = CustomEnv(ego_vehicle='car1')
        env.reset()
        return env
    time.sleep(1)
    return _init

if __name__=="__main__":

    num_trials = 4
    num_cpu = 2
    for i in range(num_trials):
        print(f"Trial: {i}")
        vec_env = CustomEnv(ego_vehicle='car1')
        #vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

        # Create log dir
        log_dir = f"tmp{i}/"
        os.makedirs(log_dir, exist_ok=True)
        vec_env = Monitor(vec_env, log_dir)

        model = PPO("MultiInputPolicy",
                    vec_env,
                    verbose=1,
                    policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                                       net_arch=[500, 300, 100],),
                    device="cuda:1")
        model.learn(total_timesteps=500_000, progress_bar=True)
        model.save(f"carla-lanefollow-empty_trial{i}")

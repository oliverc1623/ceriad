import gymnasium as gym
import racecar_gym.envs.gym_api
import numpy as np
from gymnasium import spaces


class CustomEnvRaceCar(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human", "rgb_array_follow"]}

    def __init__(self, id, scenario, render_mode):
        super().__init__()
        self.env = gym.make(
            id=id,
            scenario=scenario,
            render_mode=render_mode,
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype="float32")
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, 128, 128), dtype=np.uint8
        )

    def step(self, action):
        action_dict = {"motor": action[0], "steering": action[1]}
        obs, rewards, done, truncated, info = self.env.step(action_dict)
        return np.transpose(obs["rgb_camera"], (2,0,1)), rewards, done, truncated, info

    def reset(self, seed=None, options=dict(mode="grid")):
        obs, info = self.env.reset(seed=seed, options=options)
        return np.transpose(obs["rgb_camera"], (2,0,1)), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

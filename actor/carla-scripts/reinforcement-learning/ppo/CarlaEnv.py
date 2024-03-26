import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import random
from macad_gym.envs import MultiCarlaEnv
from skimage.color import rgb2gray

"""
    This is a scenario extracted from Carla/scenario_runner (https://github.com/carla-simulator/scenario_runner/blob/master/srunner/scenarios/follow_leading_vehicle.py).

    The configuration below contains everything you need to customize your own scenario in Macad-Gym.
"""
configs = {
    "scenarios": {
        "map": "Town01",
        "actors": {
            "car1": {
                "start": [107, 133, 0.5],
                "end": [300, 133, 0.5],
            },
        },
        "num_vehicles": 0,
        "num_pedestrians": 0,
        "weather_distribution": [0],
        "max_steps": 500
    },
    "env": {
        "server_map": "/Game/Carla/Maps/Town01",
        "render": True,
        "render_x_res": 800,    # For both Carla-Server and Manual-Control
        "render_y_res": 600,
        "x_res": 168,            # Used for camera sensor view size
        "y_res": 80,
        "framestack": 2,
        "discrete_actions": False,
        "squash_action_logits": False,
        "verbose": False,
        "use_depth_camera": False,
        "send_measurements": False,
        "enable_planner": True,
        "spectator_loc": [107, 133, 0.5],
        "sync_server": True,
        "fixed_delta_seconds": 0.05,
    },
    "actors": {
        "car1": {
            "type": "vehicle_4W",
            "enable_planner": True,
            "convert_images_to_video": False,
            "early_terminate_on_collision": True,
            "reward_function": "corl2017",
            "scenarios": "FOLLOWLEADING_TOWN1_CAR1",

            # When "auto_control" is True,
            # starts the actor using auto-pilot.
            # Allows manual control take-over on
            # pressing Key `p` on the PyGame window
            # if manual_control is also True
            "manual_control": False,
            "auto_control": False,

            "camera_type": "rgb",
            "collision_sensor": "on",
            "lane_sensor": "on",
            "log_images": False,
            "log_measurements": False,
            "render": True,
            "x_res": 168,    # Deprecated, kept for backward compatibility
            "y_res": 80,
            "use_depth_camera": False,
            "send_measurements": False,
        },
    },
}


class FollowLeadingVehicle(MultiCarlaEnv):
    """A two car Multi-Agent Carla-Gym environment"""

    def __init__(self):
        self.configs = configs
        super(FollowLeadingVehicle, self).__init__(self.configs)


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, ego_vehicle):
        super().__init__()
        self.ego_vehicle = ego_vehicle
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0,high=255,
                                 shape=(1,160,168),dtype=np.uint8),
            'throttle': spaces.Box(low=-1.0,high=1.0,
                                 shape=(1,),dtype=np.float32),
            'steer': spaces.Box(low=-1.0,high=1.0,
                                 shape=(1,),dtype=np.float32),
            'prev_steer': spaces.Box(low=-1.0,high=1.0,
                                 shape=(1,),dtype=np.float32),
        })
        self.prev_steer = np.array([0.0],dtype=np.float32)
        self.env = FollowLeadingVehicle()
        self.done = {"__all__": False}

    def step(self, action):
        action_dict = {self.ego_vehicle: action}
        observation, reward, done, info = self.env.step(action_dict)
        observation = observation[self.ego_vehicle]
        grayscale = (rgb2gray(observation) * 255).astype(np.uint8)
        grayscale = np.expand_dims(grayscale, 0)
        obs = {
            'image': grayscale,
            'throttle': np.array([action[0]],dtype=np.float32),
            'steer': np.array([action[1]],dtype=np.float32),
            'prev_steer': self.prev_steer,
        }
        self.prev_steer[0]=action[1]
        reward = reward[self.ego_vehicle]
        done = done[self.ego_vehicle]
        truncated = False
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        obs = {}
        observation = self.env.reset()
        observation = observation[self.ego_vehicle]
        grayscale = (rgb2gray(observation) * 255).astype(np.uint8)
        grayscale = np.expand_dims(grayscale, 0)
        obs['image'] = grayscale
        obs['throttle'] = np.array([0.0],dtype=np.float32)
        obs['steer'] = np.array([0.0],dtype=np.float32)
        obs['prev_steer'] = np.array([0.0],dtype=np.float32)
        self.prev_steer=np.array([0.0],dtype=np.float32)
        info = {}
        return obs, info

    def render(self):
        pass

    def close(self):
        self.env.close()

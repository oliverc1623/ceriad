'''

import gymnasium as gym
import imageio

# Initialize environment with the appropriate render mode
env = gym.make("highway-v0", render_mode="rgb_array")

# Initialize a list to hold the frames
frames = []

# Loop as usual
obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    # Render the environment to an RGB array and append to frames
    frame = env.render()
    frames.append(frame)

# Once done, write the frames to a video file
imageio.mimsave('highway_env3.gif', frames, fps=20) # For GIF

# Don't forget to close the environment
env.close()

'''

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# Initialize the environment
env = gym.make("highway-v0", render_mode='rgb_array')

# Wrap the env with a RecordVideo wrapper to enable video recording
env = RecordVideo(env, video_folder=".",
                  episode_trigger=lambda e: True,  # Record every episode
                  name_prefix="highway-v0-episode")  # Prefix for video files

# Loop through the environment
obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action = env.action_space.sample()  # Sample an action
    obs, reward, done, truncated, info = env.step(action)  # Take a step
    env.render()  # Optional for some environments

# Close the environment
env.close()

import gymnasium as gym
from matplotlib import pyplot as plt

env = gym.make('highway-v0', render_mode='rgb_array')
env.reset()

#render the environment at each step, idling for all steps
for _ in range(3):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

# Get the image from the environment and display it using matplotlib
image = env.render()
plt.imshow(image)
plt.show()

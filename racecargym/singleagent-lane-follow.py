from time import sleep
import gymnasium as gym
import racecar_gym.envs.gym_api
import matplotlib.pyplot as plt
from IPython import display

# For custom scenarios:
env = gym.make(
    id='SingleAgentRaceEnv-v0', 
    scenario='rgb-camera-scenario.yaml',
    render_mode= 'rgb_array_follow', # 'rgb_array_follow', 
)

obs, info = env.reset(seed=47, options=dict(mode='grid'))
fig,ax = plt.subplots(1,1)
myobj = ax.imshow(obs['rgb_camera'])
done = False
t = 0
while not done:
    action = env.action_space.sample()
    print(f"action: {action}")
    action['motor'] = 1
    obs, rewards, done, truncated, states = env.step(action)
    print(f"rewards: {rewards}")

    myobj.set_data(obs['rgb_camera'])
    fig.canvas.draw_idle()
    plt.pause(.01)
    t+=1

env.close()
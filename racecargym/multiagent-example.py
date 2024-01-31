from time import sleep
import gymnasium as gym
import racecar_gym.envs.gym_api
import matplotlib.pyplot as plt
from IPython import display

# For custom scenarios:
env = gym.make(
    id='MultiAgentRaceEnv-v0', 
    scenario='my-envs.yaml',
    render_mode= 'rgb_array_follow', # 'rgb_array_follow', 
)

obs, info = env.reset(options=dict(mode='grid'))
fig,ax = plt.subplots(1,1)
myobj = ax.imshow(obs['A']['rgb_camera'])
done = {'A': False, 'B': False, 'C': False, 'D': False}
t = 0
while not done['A']:
    action = env.action_space.sample()
    print(f"action: {action}")
    action['A']['motor'] = .5
    action['B']['motor'] = .9
    action['C']['motor'] = .75
    action['D']['motor'] = .9
    obs, rewards, done, truncated, states = env.step(action)
    print(f"rewards: {rewards}")

    myobj.set_data(obs['A']['rgb_camera']) # myobj.set_data(obs['A']['rgb_camera'])
    fig.canvas.draw_idle()
    plt.pause(.01)
    t+=1

env.close()
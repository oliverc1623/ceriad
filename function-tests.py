import gymnasium as gym

#make our environment
env = gym.make('highway-v0', render_mode='human')

#reset the environment
obs, info = env.reset()
done = truncated = False

#Discrete action space
ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

print(ACTIONS_ALL)

'''
#Basic loop to get availableActions from the environment, idling for each step
while not (done or truncated):
    action = env.action_type.actions_indexes["IDLE"] # Your agent code here
    availableActions = env.get_available_actions()
    print(availableActions)
    obs, reward, done, truncated, info = env.step(action)

'''

#Basic loop, taking a sample of the action space for 10 steps
for _ in range(10):
    action = env.action_space.sample()
    availableActions = env.get_available_actions()
    print(availableActions)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
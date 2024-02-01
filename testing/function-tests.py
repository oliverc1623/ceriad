import gymnasium as gym
import warnings


#ignore deprecation warnings
warnings.filterwarnings('ignore')

#make our environment
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": True,
        "order": "sorted"
    }
}
env = gym.make('highway-v0', render_mode='human')
env.configure(config)

#reset the environment
obs, info = env.reset()
done = truncated = False

#Print discrete action space
ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

print(ACTIONS_ALL)

#Print observation space
print(obs)

'''
#Basic loop to get available actions from the environment, idling for each step
while not (done or truncated):
    action = env.action_type.actions_indexes["IDLE"] # Your agent code here
    availableActions = env.get_available_actions()
    print(availableActions)
    obs, reward, done, truncated, info = env.step(action)

#reset the environment
obs, info = env.reset()
done = truncated = False 
'''

'''
#Basic loop, taking a sample of the action space for 10 steps
#Available actions output correspondes to ACTIONS_ALL dictionary
for _ in range(10):
    action = env.action_space.sample()
    availableActions = env.get_available_actions()
    print(availableActions)
    obs, reward, done, truncated, info = env.step(action)
    env.render()

#reset the environment
obs, info = env.reset()
done = truncated = False
'''

#Similar basic loop, but this time we'll be printing state space info for every step
for _ in range(10):
    
    #sample a random action each step
    action = env.action_space.sample()
    
    #use our env method to get available actions
    availableActions = env.get_available_actions()
    print(f"Available Actions: {availableActions}")

    #get our current car's x velocity
    x_vel = obs[0, 3] #normalized default
    x_vel_unnormalized = x_vel * 20
    print(f"x Velocity: {x_vel_unnormalized}")

    #get our current car's y velocity
    y_vel = obs[0, 4]
    y_vel_unnormalized = y_vel * 20
    print(f"y Velocity: {y_vel_unnormalized}")

    #get our car's current x and y coordinates in the graph
    x_pos = obs[0, 1]
    x_pos_unnormalized = x_pos * 100
    y_pos = obs[0, 2]
    y_pos_unnormalized = y_pos * 100
    print(f"(x, y) coordinates: ({x_pos_unnormalized}, {y_pos_unnormalized})")

    #print our car's current lane
    lane_id = round(y_pos_unnormalized / 4.0)
    print(f"our current lane is: {lane_id}")

    #Now, let's print the same info, but iterating through the other vehicles in our observation space
    #Our current observation space has 5 rows, meaning we have 5 cars in total. Our ego vehicle is row 0, other vehicles are rows 1 - 4
    for i in range(1, len(obs)):
        #all values here will be unnormalized
        x_position = obs[i, 1] * 100
        y_position = obs[i, 2] * 100
        x_velocity = obs[i, 3] * 20
        y_velocity = obs[i, 4] * 20
        laneid = round(y_position / 4.0)
        print(f"Vehicle {i}: X Position: {x_position}, Y Position: {y_position}, X Velocity: {x_velocity}, Y Velocity: {y_velocity}, Current Lane: {laneid}")

    obs, reward, done, truncated, info = env.step(action)
    env.render()
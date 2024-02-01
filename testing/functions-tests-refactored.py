import gymnasium as gym
import matplotlib.pyplot as plt
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
        "absolute": False,
        "order": "sorted"
    }
}
env = gym.make('highway-v0', render_mode='rgb_array')
env.configure(config)

#reset the environment
obs, info = env.reset()
done = truncated = False

frame_number = 0 #for saving frame images

#loop through 10 steps, sample random action at each step
for _ in range(10):
    
    #sample a random action each step
    action = env.action_space.sample()
    
    availableActions = env.get_available_actions()
    print(f"Ego vehicle's vailable Actions: {availableActions}")

    #get velocities
    x_vel = obs[0, 3] #normalized default
    x_vel_unnormalized = x_vel * 20
    y_vel = obs[0, 4]
    y_vel_unnormalized = y_vel * 20

    #get our car's current x and y coordinates in the graph
    x_pos = obs[0, 1]
    x_pos_unnormalized = x_pos * 100
    y_pos = obs[0, 2]
    y_pos_unnormalized = y_pos * 100

    #print our car's current lane
    lane_id = round(y_pos_unnormalized / 4.0) + 1

    print(f"Ego vehicle: X Position: {x_pos_unnormalized:.2f}, Y Position: {y_pos_unnormalized:.2f}, X Velocity: {x_vel_unnormalized:.2f}, Y Velocity: {y_vel_unnormalized:.2f}, Current Lane: {lane_id}")

    #Now, let's print the same info, but iterating through the other vehicles in our observation space
    #Our current observation space has 5 rows, meaning we have 5 cars in total. Our ego vehicle is row 0, other vehicles are rows 1 - 4
    for i in range(1, len(obs)):
        #all values here will be unnormalized
        x_position = obs[i, 1] * 100
        y_position = obs[i, 2] * 100
        x_velocity = obs[i, 3] * 20
        y_velocity = obs[i, 4] * 20
        laneid = lane_id + round(y_position / 4.0) #relative lanes
        print(f"Vehicle {i}: X Position: {x_position:.2f}, Y Position: {y_position:.2f}, X Velocity: {x_velocity:.2f}, Y Velocity: {y_velocity:.2f}, Current Lane: {laneid}")

    #env.render()
    img = env.render()
    plt.imsave(f"../test-frames/frame_{frame_number:003}.png", img)
    frame_number += 1
    obs, reward, done, truncated, info = env.step(action)
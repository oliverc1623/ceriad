import gymnasium as gym
import warnings
import matplotlib.pyplot as plt

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
env.configure({
    "action": {
        "type": "ContinuousAction"
    }
})
env.reset()

#reset the environment
obs, info = env.reset()
done = truncated = False

#Print continuous action space
print(f"Action space: {env.action_space}")

#Print observation space
print(obs)

img = env.render()
plt.imsave(f"frames/frame_{0:003}.png", img)

prompt = ""

#Similar basic loop, but this time we'll be printing state space info for every step
for i in range(1,11):
    
    #sample a random action each step
    action = env.action_space.sample()
    action[1] = 0.01

    #use our env method to get available actions
    throttle = action[0]
    steering = action[1]
    print(f"Throttle: {throttle}")
    print(f"Steering: {steering}")

    #get our current car's x velocity
    x_vel = obs[0, 3] #normalized default
    x_vel_unnormalized = x_vel * 20
    print(f"x Velocity: {x_vel_unnormalized}")

    #get our current car's y velocity
    y_vel = obs[0, 4]
    y_vel_unnormalized = y_vel * 20
    print(f"y Velocity: {y_vel}")

    #get our car's current x and y coordinates in the graph
    x_pos = obs[0, 1]
    x_pos_unnormalized = x_pos * 100
    y_pos = obs[0, 2]
    y_pos_unnormalized = y_pos * 100
    print(f"(x, y) coordinates: ({x_pos_unnormalized}, {y_pos_unnormalized})")

    #print our car's current lane
    lane_id = round(y_pos_unnormalized / 4.0) + 1
    print(f"our current lane is: {lane_id}")

    if i == 1:
        prompt += "The initial state is the following: "
    prompt += f"Here are the ego agent's available action space for throttle and steering commands: {env.action_space}. "
    prompt += f"Ego vehicle: X Position: {x_pos_unnormalized:.2f}, Y Position: {y_pos_unnormalized:.2f}, X Velocity: {x_vel_unnormalized:.2f}, Y Velocity: {y_vel_unnormalized:.2f}, Current Lane: {lane_id}. The other vehicles are: "

    #Now, let's print the same info, but iterating through the other vehicles in our observation space
    #Our current observation space has 5 rows, meaning we have 5 cars in total. Our ego vehicle is row 0, other vehicles are rows 1 - 4
    for j in range(1, len(obs)):
        #all values here will be unnormalized
        x_position = obs[j, 1] * 100
        y_position = obs[j, 2] * 100
        x_velocity = obs[j, 3] * 20
        y_velocity = obs[j, 4] * 20
        laneid = lane_id + round(y_position / 4.0)
        prompt += f" Vehicle {j}: X Position: {x_pos_unnormalized + x_position:.2f}, Y Position: {y_pos_unnormalized + y_position:.2f}, X Velocity: {x_velocity:.2f}, Y Velocity: {y_velocity:.2f}, Current Lane: {laneid}."
    
    prompt += " What should the ego agent's next task be?"
    print(prompt)
    obs, reward, done, truncated, info = env.step(action)
    img = env.render()
    plt.imsave(f"frames/frame_{i:003}.png", img)
    prompt = ""
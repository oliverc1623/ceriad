from openai import OpenAI
import gymnasium as gym
import matplotlib.pyplot as plt
import warnings

'''
This code steps through the HighwayEnv environment, integrating the observation space with the OpenAI API
I use specific helper functions to implement logic for calling the API
We step through the environment having GPT decide our next action under specific scenarios
'''

# Ignore deprecation warnings
warnings.filterwarnings('ignore')

# OpenAI ChatCompletions API client initialization

client = OpenAI(
    api_key=""
)

# Select action function, calls the ChatCompletions endpoint

def selectAction(prompt):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "Your goal is to drive safely and efficiently. You are directing the ego vehicle in our simulation, selecting actions when prompted. Respond with the action name only, and nothing else."},
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message.content.strip()

# Helper functions to check if we need to select actions other than "IDLE"

def is_too_close(obs, threshold=20):  # Adjusted threshold to 20
    """Check if any other vehicle is within a certain distance in meters."""
    for i in range(1, len(obs)):
        distance = abs(obs[0, 1] - obs[i, 1]) * 100  # Assuming the distance needs to be calculated from the ego vehicle
        if distance < threshold:
            return True
    return False

def is_off_center(y_pos, lane_width=4.0):
    """Check if the ego vehicle is significantly off-center in its lane."""
    lane_center = (round(y_pos / lane_width) * lane_width) + (lane_width / 2)
    off_center_threshold = lane_width * 0.25  # 25% of the lane width as tolerance
    return abs(y_pos - lane_center) > off_center_threshold

def is_speed_below_target(x_vel, target_speed=20):  # Adjusted function name for clarity
    """Check if the ego vehicle's speed is below the target speed."""
    return x_vel < target_speed  # Adjust as necessary

# HighwayEnv config & setup
# Our config for the simulator

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

# Reset the environment

obs, info = env.reset()
done = truncated = False

# Discrete action space

ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

# Loop through environment, sample random action at each step
while not (done or truncated):
    
    prompt = ""

    availableActions = env.get_available_actions()
    # Translate action numbers to names using ACTIONS_ALL
    availableActionNames = [ACTIONS_ALL[action] for action in availableActions]
    prompt += f"Ego vehicle's available Actions: {availableActionNames}\n"

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

    prompt += f"Ego vehicle: X Position: {x_pos_unnormalized:.2f}, Y Position: {y_pos_unnormalized:.2f}, X Velocity: {x_vel_unnormalized:.2f}, Y Velocity: {y_vel_unnormalized:.2f}, Current Lane: {lane_id}\n"

    #Now, let's print the same info, but iterating through the other vehicles in our observation space
    #Our current observation space has 5 rows, meaning we have 5 cars in total. Our ego vehicle is row 0, other vehicles are rows 1 - 4
    for i in range(1, len(obs)):
        #all values here will be unnormalized, also added to our ego vehicle to make things absolute
        x_position = x_pos_unnormalized + (obs[i, 1] * 100)
        y_position = y_pos_unnormalized + (obs[i, 2] * 100)
        x_velocity = x_vel_unnormalized + (obs[i, 3] * 20)
        y_velocity = y_vel_unnormalized + (obs[i, 4] * 20)
        laneid = round(y_position / 4.0) + 1 #relative lanes
        prompt += f"Vehicle {i}: X Position: {x_position:.2f}, Y Position: {y_position:.2f}, X Velocity: {x_velocity:.2f}, Y Velocity: {y_velocity:.2f}, Current Lane: {laneid}\n"

    print(prompt) # Print our completed prompt at each step

    # Logic to check if we need to select an action other than "IDLE"

    # Check conditions
    too_close = is_too_close(obs)
    off_center = is_off_center(y_pos_unnormalized)
    speed_below_target = is_speed_below_target(x_vel_unnormalized)

    # Decision logic
    if not too_close and speed_below_target:
        action = env.action_type.actions_indexes["FASTER"]
    elif too_close or off_center:
        action = env.action_type.actions_indexes[selectAction(prompt)]  # Call API for action decision
    else:
        action = env.action_type.actions_indexes["IDLE"]
    
    obs, reward, done, truncated, info = env.step(action)
    env.render()

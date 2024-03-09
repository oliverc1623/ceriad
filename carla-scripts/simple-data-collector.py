import glob
import os
import sys
import time
import json
import datetime

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
import logging
import random

# semantic segmentation semantic_tags
{
    0: "Unlabeled",
    1: "Roads",
    2: "SideWalks",
    3: "Building",
    4: "Wall",
    5: "Fence",
    6: "Pole",
    7: "TrafficLight",
    8: "TrafficSign",
    9: "Vegetation",
    10: "Terrain",
    11: "Sky",
    12: "Pedestrian",
    13: "Rider",
    14: "Car",
    15: "Truck",
    16: "Bus",
    17: "Train",
    18: "Motorcycle",
    19: "Bicycle",
    20: "Static",
    21: "Dynamic",
    22: "Other",
    23: "Water",
    24: "RoadLine",
    25: "Ground",
    26: "Bridge",
    27: "RailTrack",
    28: "GuardRail"
}

# Ensure the output directory exists
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# File path for JSON data
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
data_file_path = f'{output_dir}/data_{timestamp}.json'

# Open the file in append mode
with open(data_file_path, 'a') as data_file:
    data_file.write('[')  # Start of JSON array

# Global variable to hold the latest obstacle detection data
latest_obstacle_data = {}

# Function to append data to the JSON file
def append_data_to_json(data):
    with open(data_file_path, 'a') as data_file:
        if data_file.tell() > 1:  # If file is not empty, add a comma
            data_file.write(',')
        json.dump(data, data_file, indent=4, separators=(',',': '))

# Function to finalize the JSON file
def finalize_json_file():
    with open(data_file_path, 'a') as data_file:
        data_file.write(']')  # End of JSON array

def map_to_mph(value, min_mph=0, max_mph=100):
    """Maps a value from [0, 1] to a specified MPH range [min_mph, max_mph]."""
    return (value - 0) * (max_mph - min_mph) / (1 - 0) + min_mph

def map_to_steering_angle(value, min_angle=-45, max_angle=45):
    """Maps a value from [-1, 1] to a specified steering angle range [min_angle, max_angle]."""
    return (value - (-1)) * (max_angle - min_angle) / (1 - (-1)) + min_angle

def format_string(s):
    print(s)
    return s.split('.')[1].replace('_', ' ')

# Function to process camera data and include obstacle data
def process_camera_data(image, ego_vehicle):
    global latest_obstacle_data
    image_path = f'{output_dir}/{image.frame}.png'
    image.save_to_disk(image_path)

    # get ego vehicle metrics
    ego_vehicle_control = ego_vehicle.get_control()
    throttle = ego_vehicle_control.throttle
    steering = ego_vehicle_control.steer
    mph = round(map_to_mph(throttle),4)
    angle = round(map_to_steering_angle(steering),4)

    # Generate hazard response
    # TODO: list objects in scene with ss tags

    # Generate gpt response for potential hazard
    if latest_obstacle_data == {}:
        hazard_response = "No obstacles detected."
    else:
        obstacle_ss_tags = latest_obstacle_data['other_actor'].semantic_tags
        print(obstacle_ss_tags)
        obstacle = latest_obstacle_data['other_actor'].type_id
        distance = round(latest_obstacle_data['distance'], 4)

        hazard_response = "The following obstaclces could be threatful:"
        formatted_obstacle = format_string(obstacle)
        hazard_response += f" {formatted_obstacle} is {distance} meters away."

    conversations = [
        {
            'from': 'human',
            'value': f'<image>\nWhat are objects worth noting in the current scenario?',
        },
        {
            'from': 'gpt',
            'value': "place holder"
        },
        {
            'from': 'human',
            'value': f'I am the driver driving at {mph} mph and steering at {angle} degrees. What are hazards in this driving scenario?',
        },
        {
            'from': 'gpt',
            'value': hazard_response
        }
    ]

    combined_data = {
        'id': image.frame,
        'file_path': image_path,
        'conversations': conversations
    }
    append_data_to_json(combined_data)

# Callback function for obstacle detector
def obstacle_callback(data):
    global latest_obstacle_data
    latest_obstacle_data = {
        'distance': data.distance,
        'other_actor': data.other_actor,
    }

# Main data collection function
def collect_data():
    # Initialize client and world
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()

    # Define the blueprint for the rgb camera and obstacle detector
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    obstacle_bp = blueprint_library.find('sensor.other.obstacle')

    # Adjust sensor settings if necessary
    camera_bp.set_attribute('image_size_x', '120')
    camera_bp.set_attribute('image_size_y', '120')
    camera_bp.set_attribute('fov', '90')
    camera_bp.set_attribute('sensor_tick','0.25')
    obstacle_bp.set_attribute('only_dynamics','True')

    # Collect data for a certain amount of time
    try:
        world.wait_for_tick(seconds=100)

        ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name','ego')
        print('\nEgo role_name is set')
        ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
        ego_bp.set_attribute('color',ego_color)
        print('\nEgo color is set')

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        #if 0 < number_of_spawn_points:
            #random.shuffle(spawn_points)
            #ego_transform = spawn_points[0]
            #ego_vehicle = world.spawn_actor(ego_bp,ego_transform)
            #print('\nEgo is spawned')
        #else:
            #logging.warning('Could not found any spawn points')

        ego_vehicle = None
        # Get the ego vehicle
        while ego_vehicle is None:
            print("Waiting for the ego vehicle...")
            time.sleep(1)
            possible_vehicles = world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == 'hero':
                    print("Ego vehicle found")
                    ego_vehicle = vehicle
                    break

        # Spawn sensors and attach to vehicle
        camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=ego_vehicle)
        obstacle_detector = world.spawn_actor(obstacle_bp, carla.Transform(), attach_to=ego_vehicle)

        # Define callbacks
        camera.listen(lambda image: process_camera_data(image, ego_vehicle))
        obstacle_detector.listen(obstacle_callback)
        # --------------
        # Place spectator on ego spawning
        # --------------
        spectator = world.get_spectator()
        spectator.set_transform(ego_vehicle.get_transform())

        ego_vehicle.set_autopilot(True, 5000)

        # --------------
        # Game loop. Prevents the script from finishing.
        # --------------
        while True:
            world_snapshot = world.wait_for_tick()

            # --------------
            # Place spectator on ego spawning
            # --------------
            spectator = world.get_spectator()
            spectator.set_transform(ego_vehicle.get_transform())

    finally:
        # Finalize JSON file
        finalize_json_file()
        print(f'Data saved to {data_file_path}')

if __name__ == '__main__':
    collect_data()

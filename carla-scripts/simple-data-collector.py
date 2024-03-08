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

# Function to process camera data and include obstacle data
def process_camera_data(image):
    global latest_obstacle_data
    image_path = f'{output_dir}/{image.frame}.png'
    image.save_to_disk(image_path)
    combined_data = {
        'frame': image.frame,
        'timestamp': image.timestamp,
        'file_path': image_path,
        'obstacle_data': latest_obstacle_data
    }
    append_data_to_json(combined_data)

# Callback function for obstacle detector
def obstacle_callback(data):
    global latest_obstacle_data
    latest_obstacle_data = {
        'distance': data.distance,
        'other_actor': data.other_actor.type_id,
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
    camera_bp.set_attribute('sensor_tick','1')
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

        if 0 < number_of_spawn_points:
            random.shuffle(spawn_points)
            ego_transform = spawn_points[0]
            ego_vehicle = world.spawn_actor(ego_bp,ego_transform)
            print('\nEgo is spawned')
        else:
            logging.warning('Could not found any spawn points')

        # Spawn sensors and attach to vehicle
        camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=ego_vehicle)
        obstacle_detector = world.spawn_actor(obstacle_bp, carla.Transform(), attach_to=ego_vehicle)

        # Define callbacks
        camera.listen(process_camera_data)
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

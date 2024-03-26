import glob
import os
import sys
import time
import json
import datetime

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import carla
import argparse
import logging
import random
import numpy as np
import math

# semantic segmentation semantic_tags
ss_tag_id_map = {
    0: "Unlabeled",
    1: "Road",
    2: "SideWalk",
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
    28: "GuardRail",
}

# Ensure the output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# File path for JSON data
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
data_file_path = f"data_{timestamp}.json"

# Open the file in append mode
with open(data_file_path, "a") as data_file:
    data_file.write("[")  # Start of JSON array

# Global variable to hold the latest data from sensors other than rgb camera
latest_obstacle_data = {}
latest_ss_image = {}


# Function to append data to the JSON file
def append_data_to_json(data):
    with open(data_file_path, "a") as data_file:
        if data_file.tell() > 1:  # If file is not empty, add a comma
            data_file.write(",")
        json.dump(data, data_file, indent=4, separators=(",", ": "))


# Function to finalize the JSON file
def finalize_json_file():
    with open(data_file_path, "a") as data_file:
        data_file.write("]")  # End of JSON array


def map_to_mph(value, min_mph=0, max_mph=100):
    """Maps a value from [0, 1] to a specified MPH range [min_mph, max_mph]."""
    return (value - 0) * (max_mph - min_mph) / (1 - 0) + min_mph


def map_to_steering_angle(value, min_angle=-45, max_angle=45):
    """Maps a value from [-1, 1] to a specified steering angle range [min_angle, max_angle]."""
    return (value - (-1)) * (max_angle - min_angle) / (1 - (-1)) + min_angle


def format_string(s):
    print(s)
    return s.split(".")[1].replace("_", " ")


def calculate_distance(location1, location2):
    """
    Calculate the Euclidean distance between two locations.
    """
    return math.sqrt(
        (location1.x - location2.x) ** 2
        + (location1.y - location2.y) ** 2
        + (location1.z - location2.z) ** 2
    )


def describe_nearby_vehicles(nearby_vehicles):
    descriptions = [
        f"There is a vehicle {distance:.2f} meters away"
        for distance in nearby_vehicles.values()
    ]

    # Join all descriptions into a single string
    description_str = ". ".join(descriptions) + "."

    if nearby_vehicles == {}:
        return ""
    else:
        return description_str


# Function to process camera data and include obstacle data
def process_camera_data(image, ego_vehicle, world):
    global latest_obstacle_data
    image_path = f"{output_dir}/{image.frame}.jpg"
    image.save_to_disk(image_path)

    # get ego vehicle metrics
    ego_vehicle_control = ego_vehicle.get_control()
    ego_location = ego_vehicle.get_location()
    throttle = ego_vehicle_control.throttle
    steering = ego_vehicle_control.steer
    mph = round(map_to_mph(throttle), 4)
    angle = round(map_to_steering_angle(steering), 4)

    vicinity_threshold = 50.0  # Define the vicinity radius in meters
    nearby_vehicles = {}
    vehicles = world.get_actors().filter("vehicle.*")

    for vehicle in vehicles:
        if vehicle.id != ego_vehicle.id:  # Avoid calculating distance to itself
            vehicle_location = vehicle.get_location()
            distance = calculate_distance(ego_location, vehicle_location)

            forward_vec = ego_vehicle.get_transform().get_forward_vector()
            ray = (
                vehicle.get_transform().location - ego_vehicle.get_transform().location
            )

            if forward_vec.dot(ray) > 1:
                if distance <= vicinity_threshold:
                    nearby_vehicles[vehicle.id] = distance

    print(nearby_vehicles)
    hazard_response = ""
    hazard_response += describe_nearby_vehicles(nearby_vehicles)

    # Generate gpt response for potential hazard
    if latest_obstacle_data == {}:
        hazard_response += " No direct obstacles detected."
    else:
        obstacle_ss_tag_id = latest_obstacle_data["other_actor"].semantic_tags
        ss_tag = ss_tag_id_map[obstacle_ss_tag_id[0]]
        distance = round(latest_obstacle_data["distance"], 4)
        hazard_response = "The following obstaclces could be threatful:"
        hazard_response += f" a {ss_tag} is {distance} meters away."

    conversations = [
        {
            "from": "human",
            "value": f"<image>\nI am the driver driving at {mph} mph and steering at {angle} degrees. What are the vehicles in the image, their positions, and are they a possible threat to the ego vehicle based on their driving?",
        },
        {"from": "gpt", "value": hazard_response},
    ]

    combined_data = {
        "id": image.frame,
        "file_path": image_path,
        "conversations": conversations,
    }
    append_data_to_json(combined_data)
    latest_obstacle_data = {}


# Callback function for obstacle detector


def obstacle_callback(data):
    global latest_obstacle_data
    latest_obstacle_data = {
        "distance": data.distance,
        "other_actor": data.other_actor,
    }


# Callback function for ss camera
def ss_callback(ss_image):
    global latest_ss_image
    image_data = np.frombuffer(ss_image.raw_data, dtype=np.dtype("uint8"))
    image_data = np.reshape(
        image_data, (ss_image.height, ss_image.width, 4)
    )  # RGBA format
    # The object class is encoded in the red channel.
    semantic_image = image_data[:, :, 2]  # Using the red channel for class information

    num_none = np.sum(semantic_image == 0)
    num_roads = np.sum(semantic_image == 1)
    num_sidewalk = np.sum(semantic_image == 2)
    num_building = np.sum(semantic_image == 3)
    num_wall = np.sum(semantic_image == 4)
    num_fence = np.sum(semantic_image == 5)
    num_pole = np.sum(semantic_image == 6)
    num_trafficlight = np.sum(semantic_image == 7)
    num_trafficsign = np.sum(semantic_image == 8)
    num_vegetation = np.sum(semantic_image == 9)
    num_terrian = np.sum(semantic_image == 10)
    num_sky = np.sum(semantic_image == 11)
    num_pedestrian = np.sum(semantic_image == 12)
    num_rider = np.sum(semantic_image == 13)
    num_car = np.sum(semantic_image == 14)
    num_truck = np.sum(semantic_image == 15)
    num_bus = np.sum(semantic_image == 16)
    num_train = np.sum(semantic_image == 17)
    num_motorcycle = np.sum(semantic_image == 18)
    num_bicycle = np.sum(semantic_image == 19)
    num_static = np.sum(semantic_image == 20)
    num_dynamic = np.sum(semantic_image == 21)
    num_other = np.sum(semantic_image == 22)
    num_water = np.sum(semantic_image == 23)
    num_RoadLine = np.sum(semantic_image == 24)
    num_Ground = np.sum(semantic_image == 25)
    num_bridge = np.sum(semantic_image == 26)
    num_railtrack = np.sum(semantic_image == 27)
    num_guardrail = np.sum(semantic_image == 28)
    objects = {
        "None": num_none,
        "Road": num_roads,
        "Sidewalk": num_sidewalk,
        "Building": num_building,
        "Wall": num_wall,
        "Fence": num_fence,
        "Pole": num_pole,
        "Traffic light": num_trafficlight,
        "Traffic sign": num_trafficsign,
        "Terrian": num_terrian,
        "Pedestrian": num_pedestrian,
        "Rider": num_rider,
        "Car": num_car,
        "Truck": num_truck,
        "Bus": num_bus,
        "Train": num_train,
        "Motorcycle": num_motorcycle,
        "Bicycle": num_bicycle,
        "Static": num_static,
        "Dynamic": num_dynamic,
        "Water": num_water,
        "Roadline": num_RoadLine,
        "Ground": num_Ground,
        "Bridge": num_bridge,
        "Rail track": num_railtrack,
        "Gaurdrail": num_guardrail,
    }
    filtered_values = {k: v for k, v in objects.items() if v > 500}
    latest_ss_image = {
        "seen_objects": ", ".join([f"{k}" for k, v in filtered_values.items()])
    }


# Spawn spectator camera
def calculate_spectator_transform(
    vehicle_transform, offset=(-10, 0, 5), rotation_offset=(0, 0, 0)
):
    """
    Calculate the spectator transform based on the vehicle transform.
    :param vehicle_transform: The transform of the vehicle.
    :param offset: A tuple of (x, y, z) offsets relative to the vehicle.
    :param rotation_offset: A tuple of (pitch, yaw, roll) rotation offsets.
    :return: The calculated transform for the spectator.
    """
    # Extract the vehicle location and rotation
    vehicle_location = vehicle_transform.location
    vehicle_rotation = vehicle_transform.rotation
    # Convert the vehicle rotation to radians
    pitch_rad = math.radians(vehicle_rotation.pitch)
    yaw_rad = math.radians(vehicle_rotation.yaw)
    roll_rad = math.radians(vehicle_rotation.roll)
    # Calculate the forward vector
    cos_pitch, cos_yaw = math.cos(pitch_rad), math.cos(yaw_rad)
    sin_pitch, sin_yaw = math.sin(pitch_rad), math.sin(yaw_rad)
    fwd_vector = carla.Vector3D(
        x=cos_pitch * cos_yaw, y=cos_pitch * sin_yaw, z=sin_pitch
    )
    # Calculate the new location using the forward vector
    offset_location = carla.Location(
        x=vehicle_location.x + offset[0] * fwd_vector.x,
        y=vehicle_location.y + offset[0] * fwd_vector.y,
        z=vehicle_location.z + offset[2],
    )
    # Apply rotation offset to the vehicle rotation
    spectator_rotation = carla.Rotation(
        pitch=vehicle_rotation.pitch + rotation_offset[0],
        yaw=vehicle_rotation.yaw + rotation_offset[1],
        roll=vehicle_rotation.roll + rotation_offset[2],
    )
    # Create the spectator transform
    spectator_transform = carla.Transform(offset_location, spectator_rotation)
    return spectator_transform


# Main data collection function
def collect_data():
    # Initialize client and world
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)
    world = client.get_world()

    # Define the blueprint for the rgb camera and obstacle detector
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    obstacle_bp = blueprint_library.find("sensor.other.obstacle")
    ss_bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")

    # Adjust sensor settings if necessary
    camera_bp.set_attribute("image_size_x", "120")
    camera_bp.set_attribute("image_size_y", "120")
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("sensor_tick", "1")
    obstacle_bp.set_attribute("only_dynamics", "True")
    obstacle_bp.set_attribute("distance", "5")
    obstacle_bp.set_attribute("hit_radius", "1.0")

    # Collect data for a certain amount of time
    try:
        world.wait_for_tick(seconds=60)
        ego_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        ego_bp.set_attribute("role_name", "ego")
        print("\nEgo role_name is set")
        ego_color = random.choice(ego_bp.get_attribute("color").recommended_values)
        ego_bp.set_attribute("color", ego_color)
        print("\nEgo color is set")
        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        # Uncomment if using a random world
        if 0 < number_of_spawn_points:
            random.shuffle(spawn_points)
            ego_transform = spawn_points[0]
            ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
            print("\nEgo is spawned")
        else:
            logging.warning("Could not found any spawn points")
        # Spawn sensors and attach to vehicle
        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.5, z=2.0)),
            attach_to=ego_vehicle,
        )
        obstacle_detector = world.spawn_actor(
            obstacle_bp, carla.Transform(), attach_to=ego_vehicle
        )
        ss_camera = world.spawn_actor(
            ss_bp, carla.Transform(carla.Location(x=1.5, z=2.0)), attach_to=ego_vehicle
        )
        # Define callbacks
        camera.listen(lambda image: process_camera_data(image, ego_vehicle, world))
        obstacle_detector.listen(obstacle_callback)
        ss_camera.listen(ss_callback)
        # --------------
        # Place spectator on ego spawning
        # --------------
        # Get the ego vehicle's transform
        spectator_transform = calculate_spectator_transform(
            ego_vehicle.get_transform(),
            offset=(-6, 0, 2.5),
            rotation_offset=(-10, 0, 0),
        )
        spectator = world.get_spectator()
        spectator.set_transform(spectator_transform)

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
            spectator_transform = calculate_spectator_transform(
                ego_vehicle.get_transform(),
                offset=(-6, 0, 2.5),
                rotation_offset=(-10, 0, 0),
            )
            spectator.set_transform(spectator_transform)

    finally:
        # Finalize JSON file
        finalize_json_file()
        print(f"Data saved to {data_file_path}")
        client.stop_recorder()
        if ego_vehicle is not None:
            if camera is not None:
                camera.stop()
                camera.destroy()
            if obstacle_detector is not None:
                obstacle_detector.stop()
                obstacle_detector.destroy()
            if ss_camera is not None:
                ss_camera.stop()
                ss_camera.destroy()

            ego_vehicle.destroy()


if __name__ == "__main__":
    collect_data()

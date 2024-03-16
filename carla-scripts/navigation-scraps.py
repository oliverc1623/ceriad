#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import numpy as np
import math


# In[2]:


# Initialize client and world
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()

# Define the blueprint for the rgb camera and obstacle detector
blueprint_library = world.get_blueprint_library()
camera_bp = blueprint_library.find('sensor.camera.rgb')
obstacle_bp = blueprint_library.find('sensor.other.obstacle')
ss_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')

# Adjust sensor settings if necessary
camera_bp.set_attribute('image_size_x', '120')
camera_bp.set_attribute('image_size_y', '120')
camera_bp.set_attribute('fov', '90')
camera_bp.set_attribute('sensor_tick','1')
obstacle_bp.set_attribute('only_dynamics','True')
obstacle_bp.set_attribute('distance', '5')
obstacle_bp.set_attribute('hit_radius','1.0')


# In[10]:


# Spawn ego agent
ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
ego_bp.set_attribute('role_name','ego')
ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
ego_bp.set_attribute('color',ego_color)

spawn_points = world.get_map().get_spawn_points()
number_of_spawn_points = len(spawn_points)
print(f"Num spawn points: {number_of_spawn_points}")

if 0 < number_of_spawn_points:
    random.shuffle(spawn_points)
    ego_transform = spawn_points[0]
    ego_vehicle = world.spawn_actor(ego_bp,ego_transform)
    print('\nEgo is spawned')
else:
    logging.warning('Could not found any spawn points')


# In[11]:


# Spawn spectator camera
def calculate_spectator_transform(vehicle_transform, offset=(-10, 0, 5), rotation_offset=(0, 0, 0)):
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
    fwd_vector = carla.Vector3D(x=cos_pitch * cos_yaw, y=cos_pitch * sin_yaw, z=sin_pitch)
    
    # Calculate the new location using the forward vector
    offset_location = carla.Location(
        x=vehicle_location.x + offset[0] * fwd_vector.x,
        y=vehicle_location.y + offset[0] * fwd_vector.y,
        z=vehicle_location.z + offset[2]
    )
    
    # Apply rotation offset to the vehicle rotation
    spectator_rotation = carla.Rotation(
        pitch=vehicle_rotation.pitch + rotation_offset[0],
        yaw=vehicle_rotation.yaw + rotation_offset[1],
        roll=vehicle_rotation.roll + rotation_offset[2]
    )
    
    # Create the spectator transform
    spectator_transform = carla.Transform(offset_location, spectator_rotation)
    
    return spectator_transform

# Get the ego vehicle's transform
spectator_transform = calculate_spectator_transform(ego_vehicle.get_transform(),
                                                   offset=(-6, 0, 2.5), 
                                                    rotation_offset=(-10, 0, 0))
spectator = world.get_spectator()
spectator.set_transform(spectator_transform)


# In[12]:


ego_vehicle.set_autopilot(True, 5000)


# In[13]:


# --------------
# Game loop. Prevents the script from finishing.
# --------------
while True:
   world_snapshot = world.wait_for_tick()
   # --------------
   # Place spectator on ego spawning
   # --------------
   # Get the ego vehicle's transform
   spectator_transform = calculate_spectator_transform(ego_vehicle.get_transform(),
                                                      offset=(-6, 0, 2.5), 
                                                       rotation_offset=(-10, 0, 0))
   spectator = world.get_spectator()
   spectator.set_transform(spectator_transform)


# In[14]:


def clean_up():
    client.stop_recorder()
    if ego_vehicle is not None:
        # if ego_cam is not None:
        #     ego_cam.stop()
        #     ego_cam.destroy()
        # if ego_col is not None:
        #     ego_col.stop()
        #     ego_col.destroy()
        # if ego_lane is not None:
        #     ego_lane.stop()
        #     ego_lane.destroy()
        # if ego_obs is not None:
        #     ego_obs.stop()
        #     ego_obs.destroy()
        # if ego_gnss is not None:
        #     ego_gnss.stop()
        #     ego_gnss.destroy()
        # if ego_imu is not None:
        #     ego_imu.stop()
        #     ego_imu.destroy()
        ego_vehicle.destroy()
clean_up()


# In[ ]:





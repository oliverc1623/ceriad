import glob
import os
import sys
import time

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
import cv2
import matplotlib.pyplot as plt
import json

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:

        world = client.get_world()

        # uncomment for synchronous_mode
        #settings = world.get_settings()
        #settings.fixed_delta_seconds = 0.05
        #settings.synchronous_mode = True
        #world.apply_settings(settings)

        ego_vehicle = None
        ego_cam = None
        ego_col = None
        ego_lane = None
        ego_obs = None
        ego_gnss = None
        ego_imu = None

        # --------------
        # Start recording
        # --------------
        """
        client.start_recorder('~/tutorial/recorder/recording01.log')
        """

        # --------------
        # Spawn ego vehicle
        # --------------

        player_max_speed = 1.589
        player_max_speed_fast = 3.71

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


        # --------------
        # Add a RGB camera sensor to ego vehicle.
        # --------------
        cam_bp = None
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(120))
        cam_bp.set_attribute("image_size_y",str(120))
        cam_bp.set_attribute("fov",str(105))
        cam_bp.set_attribute("sensor_tick",str(1))
        cam_location = carla.Location(x=1, z=2.5)
        cam_transform = carla.Transform(cam_location)
        ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)

        ## initialize sensor data dictionary
        image_w = cam_bp.get_attribute("image_size_x").as_int()
        image_h = cam_bp.get_attribute("image_size_y").as_int()
        sensor_data = {'image': np.zeros((image_h, image_w, 3)),
                       'frame': 0}
        def rgb_callback(image, data_dict):
            data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            data_dict['frame'] = image.frame
        ego_cam.listen(lambda image: rgb_callback(image, sensor_data))

        # --------------
        # Add collision sensor to ego vehicle.
        # --------------
        col_bp = world.get_blueprint_library().find('sensor.other.collision')
        col_location = carla.Location(0,0,0)
        col_rotation = carla.Rotation(0,0,0)
        col_transform = carla.Transform(col_location,col_rotation)
        ego_col = world.spawn_actor(col_bp,col_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        def col_callback(colli, data_dict):
            #print("Collision detected:\n"+str(colli)+'\n')
            data_dict["collisionEvent"] = {'other_actor': colli.other_actor,
                                          'normal_impulse': colli.normal_impulse}
        ego_col.listen(lambda colli: col_callback(colli, sensor_data))

        # --------------
        # Add Lane invasion sensor to ego vehicle.
        # --------------
        lane_bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        lane_location = carla.Location(0,0,0)
        lane_rotation = carla.Rotation(0,0,0)
        lane_transform = carla.Transform(lane_location,lane_rotation)
        ego_lane = world.spawn_actor(lane_bp,lane_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        def lane_callback(lane, data_dict):
            #print("Lane invasion detected:\n"+str(lane)+'\n')
            data_dict['laneInvasion'] = lane.crossed_lane_markings
        ego_lane.listen(lambda lane: lane_callback(lane, sensor_data))

        # --------------
        # Add Obstacle sensor to ego vehicle.
        # --------------
        obs_bp = world.get_blueprint_library().find('sensor.other.obstacle')
        obs_bp.set_attribute("only_dynamics",str(False))
        obs_bp.set_attribute("sensor_tick",str(1))
        obs_location = carla.Location(0,0,0)
        obs_rotation = carla.Rotation(0,0,0)
        obs_transform = carla.Transform(obs_location,obs_rotation)
        ego_obs = world.spawn_actor(obs_bp,obs_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        def obs_callback(obs, data_dict):
            #print("Obstacle detected:\n"+str(obs)+'\n')
            data_dict['obstacleDetection'] = {'other_actor': obs.other_actor,
                                              'distance': obs.distance}
        ego_obs.listen(lambda obs: obs_callback(obs, sensor_data))

        # --------------
        # Add GNSS sensor to ego vehicle.
        # --------------
        """
        gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')
        gnss_location = carla.Location(0,0,0)
        gnss_rotation = carla.Rotation(0,0,0)
        gnss_transform = carla.Transform(gnss_location,gnss_rotation)
        gnss_bp.set_attribute("sensor_tick",str(3.0))
        ego_gnss = world.spawn_actor(gnss_bp,gnss_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        def gnss_callback(gnss):
            print("GNSS measure:\n"+str(gnss)+'\n')
        ego_gnss.listen(lambda gnss: gnss_callback(gnss))
        """

        # --------------
        # Add IMU sensor to ego vehicle.
        # --------------
        """
        imu_bp = world.get_blueprint_library().find('sensor.other.imu')
        imu_location = carla.Location(0,0,0)
        imu_rotation = carla.Rotation(0,0,0)
        imu_transform = carla.Transform(imu_location,imu_rotation)
        imu_bp.set_attribute("sensor_tick",str(3.0))
        ego_imu = world.spawn_actor(imu_bp,imu_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        def imu_callback(imu):
            print("IMU measure:\n"+str(imu)+'\n')
        ego_imu.listen(lambda imu: imu_callback(imu))
        """

        # --------------
        # Place spectator on ego spawning
        # --------------
        """
        spectator = world.get_spectator()
        world_snapshot = world.wait_for_tick()
        spectator.set_transform(ego_vehicle.get_transform())
        """

        # --------------
        # Enable autopilot for ego vehicle
        # --------------
        ego_vehicle.set_autopilot(True, 5000)

        # Path to save the JSON file
        file_path = 'train.json'
        listObj = []

        # --------------
        # Game loop. Prevents the script from finishing.
        # --------------
        while True:
            world_snapshot = world.wait_for_tick()

            # --------------
            # Spectator on ego position
            # --------------
            spectator = world.get_spectator()
            spectator.set_transform(ego_vehicle.get_transform())

            image = sensor_data['image']
            frame = sensor_data['frame']
            plt.imsave(f'frames/obs_{frame:006}.jpg', image)

            ego_vehicle_control = ego_vehicle.get_control()
            throttle = ego_vehicle_control.throttle
            steering = ego_vehicle_control.steer

            obstacle = "None"
            distance = 0
            if 'obstacleDetection' in sensor_data:
                obstacleDetection = sensor_data['obstacleDetection']
                obstacle = obstacleDetection['other_actor'].semantic_tags
                print(obstacle)
                distance = obstacleDetection['distance']
                gpt_reply = f"There is a {obstacle[0]} {distance} units away."
            else:
                gpt_reply = "Empty road. No hazards."
            # Prepare the data to write to JSON
            listObj.append(
                {
                    "id": frame,  # Placeholder, as the unique ID generation is not specified
                    #image": f"obs_{step:006}.png",  # Assuming a generic image file name
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"I am driving at {throttle} and steering angle is {steering}. What are hazards in this scenario?"
                        },
                        {
                            "from": "gpt",
                            "value":gpt_reply
                        }
                    ]
                }
            )

            # Writing the filtered data to a JSON file
            with open(file_path, 'w') as json_file:
                json.dump(listObj, json_file, indent=4, separators=(',',': '))

            time.sleep(1)

    finally:
        # --------------
        # Stop recording and destroy actors
        # --------------
        client.stop_recorder()
        if ego_vehicle is not None:
            if ego_cam is not None:
                ego_cam.stop()
                ego_cam.destroy()
            if ego_col is not None:
                ego_col.stop()
                ego_col.destroy()
            if ego_lane is not None:
                ego_lane.stop()
                ego_lane.destroy()
            if ego_obs is not None:
                ego_obs.stop()
                ego_obs.destroy()
            if ego_gnss is not None:
                ego_gnss.stop()
                ego_gnss.destroy()
            if ego_imu is not None:
                ego_imu.stop()
                ego_imu.destroy()
            ego_vehicle.destroy()

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone with tutorial_ego.')

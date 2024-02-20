"""CARLA basic agent acting in the env loop."""

import argparse
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import json

from carla_gym.multi_env import MultiActorCarlaEnv, DISCRETE_ACTIONS
from carla_gym.carla_api.PythonAPI.agents.navigation.basic_agent import BasicAgent
from carla_gym.core.maps.nav_utils import get_next_waypoint

step = 0
# Path to save the JSON file
file_path = '/mnt/persistent/carla-llava-data/train.json'
listObj = []

def vehicle_control_to_action(vehicle_control, is_discrete):
    """Vehicle control object to action."""
    if vehicle_control.hand_brake:
        continuous_action = [-1.0, vehicle_control.steer]
    else:
        if vehicle_control.reverse:
            continuous_action = [vehicle_control.brake - vehicle_control.throttle, vehicle_control.steer]
        else:
            continuous_action = [vehicle_control.throttle - vehicle_control.brake, vehicle_control.steer]

    def dist(a, b):
        """Distance function."""
        return math.sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]))

    if is_discrete:
        closest_action = 0
        shortest_action_distance = dist(continuous_action, DISCRETE_ACTIONS[0])

        for i in range(1, len(DISCRETE_ACTIONS)):
            d = dist(continuous_action, DISCRETE_ACTIONS[i])
            if d < shortest_action_distance:
                closest_action = i
                shortest_action_distance = d
        return closest_action

    return continuous_action

def process_image(image):
    image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    image_data = np.reshape(image_data, (image.height, image.width, 4))  # RGBA format
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
        "Vegetation": num_vegetation,
        "Terrian": num_terrian,
        "Sky": num_sky,
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
        "Other": num_other,
        "Water": num_water,
        "Roadline": num_RoadLine,
        "Ground": num_Ground,
        "Bridge": num_bridge
    }

    filtered_values = {k: v for k, v in objects.items() if v > 10000}
    print(filtered_values)
    print(f"step: {step}")

    # Prepare the data to write to JSON
    listObj.append(
        {
            "id": step,  # Placeholder, as the unique ID generation is not specified
            "image": f"observation_{step:006}.png",  # Assuming a generic image file name
            "conversations": [
                {
                    "from": "human",
                    "value": "What are the objects worth noting in the current scenario?"
                },
                {
                    "from": "gpt",
                    "value": ", ".join([f"{k}" for k, v in filtered_values.items()])
                }
            ]
        }
    )
    # Writing the filtered data to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(listObj, json_file, indent=4, separators=(',',': '))

    return "String from segmented image"

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="CARLA Manual Control Client")
    argparser.add_argument("--xml_config_path", default="configs.xml", help="Path to the xml config file")
    argparser.add_argument("--maps_path", default="/Game/Carla/Maps/", help="Path to the CARLA maps")
    argparser.add_argument("--render_mode", default="human", help="Path to the CARLA maps")

    args = vars(argparser.parse_args())
    args["discrete_action_space"] = False
    # The scenario xml config should have "enable_planner" flag
    env = MultiActorCarlaEnv(**args, actor_render_width=224, actor_render_height=224, verbose=False)
    # otherwise for PZ AEC: env = carla_gym.env(**args)

    for _ in range(1):
        agent_dict = {}
        obs = env.reset()
        total_reward_dict = {}
        import carla
        ss_bp = env.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        #ss_bp.set_attribute('sensor_tick', '1.0')

        # agents init
        for actor_id in env.actor_configs.keys():
            # Set the goal for the planner to be 0.2 m after the destination just to be sure
            dest_loc = get_next_waypoint(env.world, env._end_pos[actor_id], 20.0)
            agent = BasicAgent(env._scenario_objects[actor_id], target_speed=40)
            agent.set_destination(dest_loc)
            agent_dict[actor_id] = agent

        camera_init_trans = carla.Transform(carla.Location(z=2.5))
        ss_bp = env.world.spawn_actor(ss_bp, camera_init_trans, attach_to=env._scenario_objects["vehicle1"])
        ss_bp.listen(process_image)

        start = time.time()
        done = env.dones
        while not done["__all__"]:
            action_dict = {}
            for actor_id, agent in agent_dict.items():
                action_dict[actor_id] = vehicle_control_to_action(agent.run_step(), env.discrete_action_space)
            obs, reward, term, trunc, info = env.step(action_dict)
            ob = (obs[actor_id] * 255).astype(np.uint8)
            plt.imsave(f"/mnt/persistent/carla-llava-data/frames/observation_{step:006}.png", ob)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
            print(":{}\n\t".join(["Step#", "rew", "ep_rew", "done{}"]).format(step, reward, total_reward_dict, done))
            step += 1
        print(f"{step / (time.time() - start)} fps")
    env.close()

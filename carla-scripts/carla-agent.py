"""CARLA basic agent acting in the env loop."""

import argparse
import math
import time
import matplotlib.pyplot as plt
import numpy as np

from carla_gym.multi_env import MultiActorCarlaEnv, DISCRETE_ACTIONS
from carla_gym.carla_api.PythonAPI.agents.navigation.basic_agent import BasicAgent
from carla_gym.core.maps.nav_utils import get_next_waypoint


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

    num_none = np.sum(semantic_image == 0)  # Assuming '8' is the class ID for vehicles
    print(f"Number of none pixels: {num_none}")

    num_roads = np.sum(semantic_image == 1)  # Assuming '8' is the class ID for vehicles
    print(f"Number of road pixels: {num_roads}")

    num_sidewalk = np.sum(semantic_image == 2)  # Assuming '10' is the class ID for vehicles
    print(f"Number of sidewalk pixels: {num_sidewalk}")

    num_building = np.sum(semantic_image == 3)  # Assuming '8' is the class ID for vehicles
    print(f"Number of building pixels: {num_building}")

    num_wall = np.sum(semantic_image == 4)  # Assuming '10' is the class ID for vehicles
    print(f"Number of wall pixels: {num_wall}")

    num_fence = np.sum(semantic_image == 5)  # Assuming '8' is the class ID for vehicles
    print(f"Number of fence pixels: {num_fence}")

    num_pole = np.sum(semantic_image == 6)  # Assuming '10' is the class ID for vehicles
    print(f"Number of pole pixels: {num_pole}")

    num_trafficlight = np.sum(semantic_image == 7)  # Assuming '10' is the class ID for vehicles
    print(f"Number of traffic light pixels: {num_trafficlight}")

    num_trafficsign = np.sum(semantic_image == 8)  # Assuming '8' is the class ID for vehicles
    print(f"Number of traffic sign pixels: {num_trafficsign}")

    num_vegetation = np.sum(semantic_image == 9)  # Assuming '8' is the class ID for vehicles
    print(f"Number of vegetation pixels: {num_vegetation}")

    num_terrian = np.sum(semantic_image == 10)  # Assuming '10' is the class ID for vehicles
    print(f"Number of terrian pixels: {num_terrian}")

    num_sky = np.sum(semantic_image == 11)  # Assuming '10' is the class ID for vehicles
    print(f"Number of sky pixels: {num_sky}")

    num_pedestrian = np.sum(semantic_image == 12)  # Assuming '10' is the class ID for vehicles
    print(f"Number of pedestrian pixels: {num_pedestrian}")

    num_rider = np.sum(semantic_image == 13)  # Assuming '10' is the class ID for vehicles
    print(f"Number of rider pixels: {num_rider}")

    num_car = np.sum(semantic_image == 14)  # Assuming '10' is the class ID for vehicles
    print(f"Number of car pixels: {num_car}")

    num_truck = np.sum(semantic_image == 15)  # Assuming '10' is the class ID for vehicles
    print(f"Number of truck pixels: {num_truck}")

    num_bus = np.sum(semantic_image == 16)  # Assuming '10' is the class ID for vehicles
    print(f"Number of bus pixels: {num_bus}")

    num_train = np.sum(semantic_image == 17)  # Assuming '10' is the class ID for vehicles
    print(f"Number of train pixels: {num_train}")

    num_motorcycle = np.sum(semantic_image == 18)  # Assuming '10' is the class ID for vehicles
    print(f"Number of motorcycle pixels: {num_motorcycle}")

    num_bicycle = np.sum(semantic_image == 19)  # Assuming '10' is the class ID for vehicles
    print(f"Number of bicycle pixels: {num_bicycle}")

    num_static = np.sum(semantic_image == 20)  # Assuming '10' is the class ID for vehicles
    print(f"Number of static pixels: {num_static}")

    num_dynamic = np.sum(semantic_image == 21)  # Assuming '10' is the class ID for vehicles
    print(f"Number of dynamic pixels: {num_dynamic}")

    num_other = np.sum(semantic_image == 22)  # Assuming '10' is the class ID for vehicles
    print(f"Number of other pixels: {num_other}")

    num_water = np.sum(semantic_image == 23)  # Assuming '10' is the class ID for vehicles
    print(f"Number of water pixels: {num_water}")

    num_RoadLine = np.sum(semantic_image == 24)  # Assuming '10' is the class ID for vehicles
    print(f"Number of road line pixels: {num_RoadLine}")

    num_Ground = np.sum(semantic_image == 25)  # Assuming '10' is the class ID for vehicles
    print(f"Number of ground pixels: {num_Ground}")

    num_bridge = np.sum(semantic_image == 26)  # Assuming '10' is the class ID for vehicles
    print(f"Number of bridge pixels: {num_bridge}")

    print("")

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

    for _ in range(2):
        agent_dict = {}
        obs = env.reset()
        total_reward_dict = {}
        import carla
        ss_bp = env.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')

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
        step = 0
        done = env.dones
        while not done["__all__"]:
            step += 1
            action_dict = {}
            for actor_id, agent in agent_dict.items():
                action_dict[actor_id] = vehicle_control_to_action(agent.run_step(), env.discrete_action_space)
            obs, reward, term, trunc, info = env.step(action_dict)
            ob = (obs[actor_id] * 255).astype(np.uint8)
            plt.imsave(f"frames/observation_{step:004}.png", ob)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
            print(":{}\n\t".join(["Step#", "rew", "ep_rew", "done{}"]).format(step, reward, total_reward_dict, done))
        print(f"{step / (time.time() - start)} fps")
    env.close()

import carla
import time
import json
import random
import argparse
import cv2
import numpy as np
from constants import const
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_walkers', type=int, default=100)
    parser.add_argument('--num_vehicles', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def spawn_vehicles(world, blueprint_library, num_spawn, exclude_list = ['vehicle.mitsubishi.fusorosa']):
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    spawned_cars = []
    while len(spawned_cars) < num_spawn:
        for i in range(num_spawn - len(spawned_cars)): # Try to spawn num_spawn vehicles
            blueprint = random.choice(vehicle_blueprints)
            while blueprint.id in exclude_list:
                blueprint = random.choice(vehicle_blueprints)
            transform = random.choice(spawn_points)
            vehicle = world.try_spawn_actor(blueprint, transform)
            if vehicle:
                vehicle.set_autopilot(True)
                spawned_cars.append(vehicle)
        print("Remaining", num_spawn - len(spawned_cars))
        world.tick()
    return spawned_cars # List of actors 

def spawn_walkers(world, blueprint_library, num_walkers):
    blueprintsWalkers = blueprint_library.filter('walker.pedestrian.*')
    percentagePedestriansRunning = 0.2
    percentagePedestriansCrossing = 0.2
    spawned_walkers = []
    spawned_controllers = []
    walker_controller_bp = blueprint_library.find('controller.ai.walker')
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    while len(spawned_walkers) < num_walkers:
        new_controllers = []
        for _ in range(num_walkers - len(spawned_walkers)):
            bp = random.choice(blueprintsWalkers)
            loc = world.get_random_location_from_navigation()
            if not loc:
                continue
            spawn_point = carla.Transform(loc)
            walker = world.try_spawn_actor(bp, spawn_point)
            if walker:
                controller = world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
                spawned_walkers.append(walker)
                new_controllers.append(controller)

        world.tick()
        for controller in new_controllers:
            controller.start()
            speed = 1.4 if random.random() > percentagePedestriansRunning else 2.0
            dest = world.get_random_location_from_navigation()
            if dest:
                controller.go_to_location(dest)
                controller.set_max_speed(speed)

        spawned_controllers.extend(new_controllers)

    return spawned_walkers, spawned_controllers

# Image is an object containing raw_data, which is a long buffer of 8 bit BGRA values
def save_image_to_disk(image, sensor_name):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3] # Ignore the A channel
    cv2.imwrite(f'./data/{str(image.frame).zfill(8)}_{sensor_name}.png', array)


def spawn_camera_sensor(camera_bp, world, sensor_name):
    camera_bp.set_attribute('image_size_x', str(const.IMAGE_WIDTH))
    camera_bp.set_attribute('image_size_y', str(const.IMAGE_HEIGHT))
    camera_bp.set_attribute('fov', '90')
    camera_bp.set_attribute('sensor_tick', '0.05')
    cam_transform = carla.Transform(
        carla.Location(x=325, y=172, z=14),  # Height to view town
        carla.Rotation(pitch=-36, yaw=85, roll=0)
    )

    camera = world.spawn_actor(camera_bp, cam_transform)
    camera.listen(lambda content: save_image_to_disk(content, sensor_name))
    return camera

def main(args):
    random.seed(args.seed)
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.global_percentage_speed_difference(99.0)

    world = client.load_world('Town01')  # Town01 is the first in CARLA
    blueprint_library = world.get_blueprint_library()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    weather = carla.WeatherParameters.ClearNoon
    world.set_weather(weather)
    try:
        all_actors = []
        vehicle_actors = spawn_vehicles(world, blueprint_library, args.num_vehicles)
        all_actors.extend(vehicle_actors)
        print(f"SPAWNED {len(vehicle_actors)}")
        walker_actors, controllers = spawn_walkers(world, blueprint_library, num_walkers=args.num_walkers)
        all_actors.extend(walker_actors)

        # Spawn the camera
        rgb_camera = spawn_camera_sensor(blueprint_library.find('sensor.camera.rgb'), world, 'rgb')
        sem_camera = spawn_camera_sensor(blueprint_library.find('sensor.camera.instance_segmentation'), world, 'sem')
        world.tick()
        extrinsic = np.array(rgb_camera.get_transform().get_inverse_matrix())
        intrinsic = utils.get_intrinsic_matrix(rgb_camera)
        camera_info = {
            'extrinsic':extrinsic.tolist(),
            'intrinsic':intrinsic.tolist()
        }
        with open('cam_info.json', 'w') as handle:
            json.dump(camera_info, handle)
        all_actors.append(rgb_camera)
        all_actors.append(sem_camera)

        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(
            carla.Location(x=325, y=172, z=14),  # Height to view town
            carla.Rotation(pitch=-36, yaw=85, roll=0)
        ))
        iter_idx = 0
        wframe_start = world.get_snapshot().frame
        w_frame = world.get_snapshot().frame
        while w_frame - wframe_start < 30000 :
            iter_idx += 1
            w_frame = world.get_snapshot().frame
            if iter_idx == 10:
                traffic_manager.global_percentage_speed_difference(0.0)
            
            result_list = []
            for actor in all_actors:
                try:
                    if world.get_actor(actor.id):
                        actor_dict = {}
                        actor_dict['id'] = actor.id
                        actor_dict['type'] = actor.type_id
                        bbox = actor.bounding_box
                        verts = [[v.x, v.y, v.z] for v in bbox.get_world_vertices(actor.get_transform())]
                        actor_dict['bbox'] = verts
                        result_list.append(actor_dict)
                except RuntimeError:
                    print(f"Actor {actor.id} has been destroyed")
            with open(f'data/{str(w_frame).zfill(8)}_gt.json', 'w') as handle:
                json.dump(result_list, handle, indent=4)
            
            world.tick()
    finally:
        print('\nCleaning up actors...')
        # 1. Disable sync mode before cleanup
        settings.synchronous_mode = False
        world.apply_settings(settings)

        # 2. Disable Traffic Manager sync
        traffic_manager.set_synchronous_mode(False)
        for controller in controllers:
            controller.stop()
        # 3. Destroy actors safely
        client.apply_batch([carla.command.DestroyActor(x) for x in all_actors])
        print('Cleanup done, exiting cleanly.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
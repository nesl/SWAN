import carla
import random
import time
import cv2
import numpy as np
import queue
from constants import const
import math

frame_iter = 0
target_object_actors = []

# Returns image
def process_image(image):
    pass
    # array = np.frombuffer(image.raw_data, dtype=np.uint8)
    # array = array.reshape((image.height, image.width, 4))[:, :, :3]
    # cv2.imwrite(f'./data/{str(frame_iter).zfill(5)}.png', array)

from collections import defaultdict
def process_sem(image):

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    cv2.imwrite(f'./data/{str(frame_iter).zfill(5)}_sem.png', array[:, :, :3]) # save this for viz
    in_frame_objects = defaultdict(int)
    for i in range(image.height):
        for j in range(image.width):
            if (array[i][j][2] in [12, 14, 15, 16, 17, 18, 19]): # BGRA
                bgr_key = str(array[i][j][0]) + str(array[i][j][1]) # BG
                in_frame_objects[bgr_key] += 1
    print(len(in_frame_objects))
    #in_frame_objects = {k:v for k, v in in_frame_objects.items() if v > 100} # Filter by > 100 keys
    # all_bbox_coords = []
    # for item in target_object_actors:
    #     actor_id = item.id
    #     G = str((actor_id & 0x00ff) >> 0)
    #     B = str((actor_id & 0xff00) >> 8)
    #     if (B + G) in in_frame_objects:
    #         # aggregate and save bounding boxes
    #         bbox = item.bounding_box
    #         verts = [[v.x, v.y, v.z] for v in bbox.get_world_vertices(item.get_transform())]
    #         all_bbox_coords.append(verts)
    
            
    # np.save(f'./data/{str(frame_iter).zfill(5)}_gt_bbox.npy', all_bbox_coords)# BGRA format

# Returns depth data in meters (save as .npy?)
def process_depth(depth):
    depth_arr = np.frombuffer(depth.raw_data, dtype=np.uint8)
    array = depth_arr.reshape((depth.height, depth.width, 4))[:, :, :3]
    distance_arr = (array[:, :, 2] + 256 * array[:, :, 1] + 256 * 256 * array[:, :, 0]) / (256 * 256 * 256 - 1)
    in_meters = distance_arr * 1000
    return in_meters

# Returns depth as rgb image that we can visualize as grayscale
def process_depth_into_rgb(depth):
    pass
    # depth_arr = np.frombuffer(depth.raw_data, dtype=np.uint8)
    # array = depth_arr.reshape((depth.height, depth.width, 4))[:, :, :3]
    # array = array.astype(float)
    # distance_arr = (array[:, :, 2] + 256 * array[:, :, 1] + 256 * 256 * array[:, :, 0]) / (256 * 256 * 256 - 1)
    # in_meters = distance_arr * 1000
    # normalized = np.clip(in_meters, 0, 100) / 100 # Limit to 100m 
    # in_meters = (normalized * 255).astype(np.uint8)
    # rgb_img = cv2.cvtColor(in_meters, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite(f'./data/{str(frame_iter).zfill(5)}_depth.png', rgb_img)


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def spawn_walkers(world, client, seed, num_walkers):
    blueprintsWalkers = get_actor_blueprints(world, 'walker.pedestrian.*', '2')
    percentagePedestriansRunning = 0.2      # how many pedestrians will run
    percentagePedestriansCrossing = 0.3     # how many pedestrians will walk through the road
    if seed:
        world.set_pedestrians_seed(seed)
        random.seed(seed)
    # 1. take all the random locations to spawn
    spawn_points = []
    for i in range(num_walkers):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    # 2. we spawn the walker object
    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    walkers_list = []
    walker_speed2 = []
    for i in range(len(results)):
        if not results[i].error:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if not results[i].error:
            walkers_list[i]["con"] = results[i].actor_id
    # 4. we put together the walkers and controllers id to get the objects from their id
    all_id = []
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)


    world.tick()

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
    
    return all_actors # con, walker



# Draw the pyramid showcasing the ground truth FOV
def draw_camera_fov(world, camera, fov_degrees=90.0, distance=10.0, color=carla.Color(0, 255, 0), life_time=0.1):
    # Get camera transform
    cam_tf = camera.get_transform()
    cam_loc = cam_tf.location
    cam_rot = cam_tf.rotation

    # Camera basis vectors
    forward_vec = cam_tf.get_forward_vector()
    right_vec = cam_tf.get_right_vector()
    up_vec = cam_tf.get_up_vector()

    # Compute half angles in radians
    half_fov = math.radians(fov_degrees / 2)
    aspect_ratio = const.IMAGE_WIDTH / const.IMAGE_HEIGHT  # modify if your camera isn't 16:9

    # Compute far plane dimensions
    half_width = distance * math.tan(half_fov)
    half_height = half_width / aspect_ratio

    # Far plane center
    far_center = cam_loc + forward_vec * distance

    # Compute corners in world space
    corners = [
        far_center + (up_vec * half_height) + (right_vec * half_width),   # top-right
        far_center + (up_vec * half_height) - (right_vec * half_width),   # top-left
        far_center - (up_vec * half_height) - (right_vec * half_width),   # bottom-left
        far_center - (up_vec * half_height) + (right_vec * half_width),   # bottom-right
    ]

    # Draw pyramid lines
    for corner in corners:
        world.debug.draw_line(cam_loc, corner, thickness=0.05, color=color, life_time=life_time)

    # Draw the base rectangle
    for i in range(4):
        world.debug.draw_line(corners[i], corners[(i+1) % 4], thickness=0.05, color=color, life_time=life_time)



def main():
    random.seed(100)
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.load_world('Town01')  # Town01 is the first in CARLA
    blueprint_library = world.get_blueprint_library()

    # Set synchronous mode (optional, can be removed for async mode)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Set weather
    weather = carla.WeatherParameters(sun_altitude_angle=90)
    world.set_weather(weather)
    # Alter the streetlights to simulate an environment where the camera is going to struggle
    lmanager = world.get_lightmanager()
    my_lights = lmanager.get_all_lights()
    lmanager.turn_off(my_lights)

    try:
        actors = []

        # Spawn autopiloted vehicles
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        spawn_points = world.get_map().get_spawn_points()
        num_spawned_vehicles = 0
        for i in range(100): # Spawn 100 vehicles
            blueprint = random.choice(vehicle_blueprints)
            transform = random.choice(spawn_points)
            vehicle = world.try_spawn_actor(blueprint, transform)
            if vehicle:
                vehicle.set_autopilot(True)
                actors.append(vehicle)
                target_object_actors.append(vehicle)
                num_spawned_vehicles += 1

        walker_controller_actors = spawn_walkers(world, client, seed=100, num_walkers=100)
        target_object_actors.extend(list(walker_controller_actors)[1::2])
        actors.extend(walker_controller_actors)
        print("SPAWN SUMMARY")
        print(f"SPAWNED {num_spawned_vehicles} VEHICLES")
        print(f"SPAWNED {len(walker_controller_actors)//2} WALKERS")

        # Set up RGB camera at a fixed location
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(const.IMAGE_WIDTH))
        camera_bp.set_attribute('image_size_y', str(const.IMAGE_HEIGHT))
        camera_bp.set_attribute('fov', '120')
        camera_bp.set_attribute('sensor_tick', '0.2')

        depth_cam = blueprint_library.find('sensor.camera.depth')
        depth_cam.set_attribute('image_size_x', str(const.IMAGE_WIDTH))
        depth_cam.set_attribute('image_size_y', str(const.IMAGE_HEIGHT))
        depth_cam.set_attribute('fov', '120')
        depth_cam.set_attribute('sensor_tick', '0.2')

        sem_cam = blueprint_library.find('sensor.camera.instance_segmentation')
        sem_cam.set_attribute('image_size_x', str(960))
        sem_cam.set_attribute('image_size_y', str(540))
        sem_cam.set_attribute('fov', '120')
        sem_cam.set_attribute('sensor_tick', '0.2')


        # Fixed location - manually set or use a known spot
        cam_transform = carla.Transform(
            carla.Location(x=196, y=187, z=9),  # Height to view town
            carla.Rotation(pitch=-44, yaw=56, roll=2.37)
        )

        camera = world.spawn_actor(camera_bp, cam_transform)
        actors.append(camera)
        depth_actor = world.spawn_actor(depth_cam, cam_transform)
        actors.append(depth_actor)
        sem_actor = world.spawn_actor(sem_cam, cam_transform)
        actors.append(sem_actor)

        depth_actor.listen(process_depth_into_rgb)
        camera.listen(process_image)
        sem_actor.listen(process_sem)

        # Run simulation
        for i in range(10000):
            world.tick()
            # global frame_iter
            # frame_iter += 1

            # spectator = world.get_spectator()
            # transform = spectator.get_transform()
            # location = transform.location
            # rotation = transform.rotation
            # print("Frame", i)
            # # print(f'Frame {i} location {location.x}, {location.y}, {location.z}')
            # # print(f'Frame {i} rotation {rotation.pitch}, {rotation.yaw}, {rotation.roll}')


    finally:
        print('\nCleaning up actors...')
        for actor in actors:
            actor.destroy()
        cv2.destroyAllWindows()
        print('Done.')

if __name__ == '__main__':
    main()

import carla
import numpy as np
from tqdm import tqdm
import cv2
import os

def get_intrinsic_matrix(sensor):
    image_w = int(sensor.attributes.get('image_size_x', 800))
    image_h = int(sensor.attributes.get('image_size_y', 600))
    fov = float(sensor.attributes.get('fov', 90))  # horizontal FOV in degrees

    # Compute focal length in pixels
    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
    cx = image_w / 2.0
    cy = image_h / 2.0

    intrinsic = np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1]
    ])

    return intrinsic



# Path to the folder containing images

def render_video_from_images(image_folder, output_video):
    # Get all image files, sorted (important!)
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # Make sure the frames are in order

    # Read the first image to get the size
    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4
    fps = 30  # frames per second
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write each image to the video
    for image in tqdm(images):
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video saved as {output_video}")




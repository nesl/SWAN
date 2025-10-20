import numpy as np
from PIL import Image
import json
import cv2
import argparse
from constants import const

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', type=str, choices=['render_vid', 'filter_bbox', 'filter_2d_bbox'], required=True)
    parser.add_argument('--start_frame', type=int, required=True)
    parser.add_argument('--end_frame', type=int, required=True)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--output_dir', type=str, default='outputs/')
    return parser.parse_args()

# World points N x 4 x 8 representing all bbox coords
def project_to_image(world_points, intrinsic, world2cam):
    """
    world_points: (4, 8)
    K: (3, 3)
    T_world2cam: (4, 4)
    Returns pixel coordinates (N, u, v)
    """
    # Transform world → camera coordinates
    cam_points = np.matmul(world2cam[None, :, :], world_points) # (N, 4, 8)
    cam_points = cam_points[:, :3, :]  # drop homogeneous

    # Reorder world points per: https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
    cam_points = cam_points[:, [1, 2, 0], :] # Rearrange 
    cam_points[:, 1, :] *= -1
    
    # Project camera → image plane
    pixels = np.matmul(intrinsic[None, :, :], cam_points) # (N, 3, 8)
    pixels /= pixels[:, 2:3, :]  # normalize by z

    pixels = pixels[:, :2, :] # u, v coordinates (N, 2, 8)

    return np.transpose(pixels, (0, 2, 1))



def get_in_fov_boxes(bbox_json_file, sem_seg_file, return_json_dataset=False):
    filtered_bbox_list = []
    with open(bbox_json_file, 'r') as handle:
        bbox_list = json.load(handle)
    sem_seg = np.array(Image.open(sem_seg_file))
    id_set = {}
    h, w, c = sem_seg.shape
    for i in range(h):
        for j in range(w):
            if sem_seg[i][j][0] in [12, 14, 15, 16, 17, 18, 19]:
                str_id = str(sem_seg[i][j][1]) + str(sem_seg[i][j][2])
                if str_id not in id_set:
                    id_set[str_id] = 0
                else:
                    id_set[str_id] += 1
    id_set = {k:v for k, v in id_set.items() if v > 40} # FILTER FOR ONLY GREATER THAN 40 PIXELS
    print(id_set)
    for entry in bbox_list:
        G = str((entry['id'] & 0x00ff) >> 0)
        B = str((entry['id'] & 0xff00) >> 8)
        if (G + B) in id_set:
            if return_json_dataset:
                filtered_bbox_list.append(entry)
            else:
                filtered_bbox_list.append(entry['bbox'])
    return filtered_bbox_list



def convert_to_coco(bbox_volume):
    bbox_volume = np.squeeze(bbox_volume)
    x_min = float(np.min(bbox_volume[:, 0]))
    y_min = float(np.min(bbox_volume[:, 1]))
    x_max = float(np.max(bbox_volume[:, 0]))
    y_max = float(np.max(bbox_volume[:, 1]))
    width = x_max - x_min
    height = y_max - y_min

    return [x_min, y_min, width, height] # x is across column to the right




def render_bbox_index(idx, args):
    str_idx = str(idx).zfill(8)
    try:
        filtered_bbox = np.array(get_in_fov_boxes(f'{args.data_dir}/{str_idx}_gt.json', f'{args.data_dir}/{str_idx}_sem.png')) # N x 8 x 3
        filtered_bbox = np.concatenate([filtered_bbox, np.ones((len(filtered_bbox), 8, 1))], axis=-1) # N x 8 x 4
        filtered_bbox = np.transpose(filtered_bbox, (0, 2, 1)) # N x 4 x 8

        with open('cam_info.json', 'r') as handle:
            cam_info = json.load(handle)
            extrinsic = np.array(cam_info['extrinsic'])
            intrinsic = np.array(cam_info['intrinsic'])
        image_bbox = project_to_image(filtered_bbox, intrinsic, extrinsic) # N

        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        image = cv2.imread(f'data/{str_idx}_rgb.png')
        selected_bbox = []
        for bbox in image_bbox:
            # Exclude top left detections
            if bbox[0][0] < 200 and bbox[0][1] < 100:
                continue
            selected_bbox.append(convert_to_coco(bbox)) # Take the first face only as the detections?
            for i, j in edges:
                pt1, pt2 = tuple(bbox[i].astype(int)), tuple(bbox[j].astype(int))
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)

        return image, selected_bbox
    
    except FileNotFoundError:
        print("Misaligned sensor data? Not all files exist for this index")
        return None


def main(args):
    if args.operation == 'render_vid':
        output_video = 'rendered.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4
        fps = 20  # frames per second
        video = cv2.VideoWriter(output_video, fourcc, fps, (const.IMAGE_WIDTH, const.IMAGE_HEIGHT))
        for idx in range(args.start_frame, args.end_frame):
            img, _ = render_bbox_index(idx, args)
            if img is not None:
                video.write(img)
        video.release()
    elif args.operation == 'filter_3d_bbox':
        for idx in range(args.start_frame, args.end_frame):
            try:
                filtered_bboxes = get_in_fov_boxes(f'{args.data_dir}/{str(idx).zfill(8)}_gt.json', f'{args.data_dir}/{str(idx).zfill(8)}_sem.png', return_json_dataset=True)
                with open(f'{args.output_dir}/{str(idx).zfill(8)}_gt.json', 'w') as handle:
                    json.dump(filtered_bboxes, handle, indent=2)
            except FileNotFoundError as e:
                print("File not found for index", idx)
                print(e)
    elif args.operation == 'filter_2d_bbox':
        with open('cam_info.json', 'r') as handle:
            cam_info = json.load(handle)
            extrinsic = np.array(cam_info['extrinsic'])
            intrinsic = np.array(cam_info['intrinsic'])
        for idx in range(args.start_frame, args.end_frame):
            try:
                filtered_bbox = get_in_fov_boxes(f'{args.data_dir}/{str(idx).zfill(8)}_gt.json', f'{args.data_dir}/{str(idx).zfill(8)}_sem.png', return_json_dataset=True) # N x 8 x 3
                new_bbox_list = []
                for entry in filtered_bbox:
                    bbox = np.array(entry['bbox'])[None, :, :]
                    bbox = np.concatenate([bbox, np.ones((1, 8, 1))], axis=-1)
                    bbox = np.transpose(bbox, (0, 2, 1)) # N x 4 x 8
                    image_coords = project_to_image(bbox, intrinsic, extrinsic)
                    if image_coords[0][0][0] < 200 and image_coords[0][0][1] < 100:
                        continue
                    entry['bbox'] = convert_to_coco(image_coords)
                    new_bbox_list.append(entry)
                with open(f'{args.output_dir}/{str(idx).zfill(8)}_filtered_gt.json', 'w') as handle:
                    json.dump(new_bbox_list, handle, indent=2)
            except FileNotFoundError as e:
                print("File not found for index", idx)
                print(e)

if __name__=='__main__':
    args = parse_args()
    main(args)

import json
import os

coco_annotations = {}
coco_annotations['type']='instances'
coco_annotations['images'] = []

coco_annotations["categories"] = [
       {
            "supercategory": "person",
            "name": "person",
            "id": 1
        },
        {
            "supercategory":'vehicle',
            "name":"vehicle",
            "id": 2
        }
]

coco_annotations['annotations'] = []



all_files = sorted(os.listdir('outputs'))
train_files = all_files[:8500]
val_files = all_files[8500:]

seq_length = len(train_files)
image_id = 0
detection_id = 0
track_id = 0
image_id_to_track_id = {}

for annotation in train_files:
    # Add image annotation
    coco_annotations['images'].append(
        {
            "file_name":annotation.split('_')[0] + '_rgb.png',
            'height':720,
            'width':1280,
            'id':image_id,
            'frame_id':image_id,
            'seq_length':seq_length,
            'first_frame_image_id':0
        }
    )

    # Add detection_annotation
    with open(os.path.join('outputs', annotation), 'r') as handle:
        frame_annotations = json.load(handle)
    for object in frame_annotations:
        curr_annotation = {}
        # Add the track_id
        if object['id'] not in image_id_to_track_id:
            image_id_to_track_id[object['id']] = track_id
            track_id += 1
        curr_annotation['track_id'] = image_id_to_track_id[object['id']] # Use the dict

        # Add the bbox
        curr_annotation['bbox'] = [int(coord) for coord in object['bbox']]

        # Add the category id
        curr_annotation['category_id'] = 2 if 'vehicle' in object['type'] else 1

        # Add the image_id, corresponds to the upper loop image_id
        curr_annotation['image_id'] = image_id

        # Add the id (just an incrementing counter)
        curr_annotation['id'] = detection_id
        detection_id += 1

        curr_annotation['segmentation'] = [] # No segmentation mask
        curr_annotation['ignore'] = 0
        curr_annotation['visibility'] = 1.0
        curr_annotation['area'] = curr_annotation['bbox'][2] * curr_annotation['bbox'][3]
        curr_annotation['iscrowd'] = 0
        curr_annotation['seq'] = "CARLA"

        coco_annotations['annotations'].append(curr_annotation)
    image_id += 1


with open('train.json', 'w') as handle:
    json.dump(coco_annotations, handle, indent=2)


coco_annotations['images'] = []
coco_annotations['annotations'] = []

seq_length = len(val_files)
image_id = 0
detection_id = 0
track_id = 0
image_id_to_track_id = {}

for annotation in val_files:
    # Add image annotation
    coco_annotations['images'].append(
        {
            "file_name":annotation.split('_')[0] + '_rgb.png',
            'height':720,
            'width':1280,
            'id':image_id,
            'frame_id':image_id,
            'seq_length':seq_length,
            'first_frame_image_id':0
        }
    )

    # Add detection_annotation
    with open(os.path.join('outputs', annotation), 'r') as handle:
        frame_annotations = json.load(handle)
    for object in frame_annotations:
        curr_annotation = {}
        # Add the track_id
        if object['id'] not in image_id_to_track_id:
            image_id_to_track_id[object['id']] = track_id
            track_id += 1
        curr_annotation['track_id'] = image_id_to_track_id[object['id']] # Use the dict

        # Add the bbox
        curr_annotation['bbox'] = [int(coord) for coord in object['bbox']]

        # Add the category id
        curr_annotation['category_id'] = 2 if 'vehicle' in object['type'] else 1

        # Add the image_id, corresponds to the upper loop image_id
        curr_annotation['image_id'] = image_id

        # Add the id (just an incrementing counter)
        curr_annotation['id'] = detection_id
        detection_id += 1

        curr_annotation['segmentation'] = [] # No segmentation mask
        curr_annotation['ignore'] = 0
        curr_annotation['visibility'] = 1.0
        curr_annotation['area'] = curr_annotation['bbox'][2] * curr_annotation['bbox'][3]
        curr_annotation['iscrowd'] = 0
        curr_annotation['seq'] = "CARLA"

        coco_annotations['annotations'].append(curr_annotation)
    image_id += 1


with open('val.json', 'w') as handle:
    json.dump(coco_annotations, handle, indent=2)



    


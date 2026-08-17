# Data Acquisition

1. Download the nuScenes dataset

2. Clone the [MultiCorrupt](https://github.com/ika-rwth-aachen/multicorrupt) repository. Follow the instrufctions to generate the following corruption types at maximum strength (i.e., 3) **beams_reducing**,**dark**,**lidar_fog**,**camera_fog**,**lidar_motionblur**,**camera_motionblur**. Be sure to enable `--sweep true` when generating data. 

3. Simlink the data in the MultiCorrupt data folder and organize folders following the tree. This allows each corruption directory (e.g., beamsreducing) to appear as a complete dataset from the perspective of the code. We use simlinks to avoid duplicate copies of nuScenes data. Use relative paths when possible to ensure it still works in the docker container. 

```text
├── beamsreducing
│   └── 3
│       ├── samples
│       │   ├── CAM_BACK -> ../../../../nuscenes/samples/CAM_BACK
│       │   ├── CAM_BACK_LEFT -> ../../../../nuscenes/samples/CAM_BACK_LEFT
│       │   ├── CAM_BACK_RIGHT -> ../../../../nuscenes/samples/CAM_BACK_RIGHT
│       │   ├── CAM_FRONT -> ../../../../nuscenes/samples/CAM_FRONT
│       │   ├── CAM_FRONT_LEFT -> ../../../../nuscenes/samples/CAM_FRONT_LEFT
│       │   ├── CAM_FRONT_RIGHT -> ../../../../nuscenes/samples/CAM_FRONT_RIGHT
│       │   └── LIDAR_TOP
│       └── sweeps
│           └── LIDAR_TOP
├── camera_fog
│   └── 3
│       ├── samples
│       │   ├── CAM_BACK
│       │   ├── CAM_BACK_LEFT
│       │   ├── CAM_BACK_RIGHT
│       │   ├── CAM_FRONT
│       │   ├── CAM_FRONT_LEFT
│       │   ├── CAM_FRONT_RIGHT
│       │   └── LIDAR_TOP -> ../../../../nuscenes/samples/LIDAR_TOP/
│       └── sweeps
│           └── LIDAR_TOP -> ../../../../nuscenes/sweeps/LIDAR_TOP/
├── camera_motionblur
│   └── 3
│       ├── samples
│       │   ├── CAM_BACK
│       │   ├── CAM_BACK_LEFT
│       │   ├── CAM_BACK_RIGHT
│       │   ├── CAM_FRONT
│       │   ├── CAM_FRONT_LEFT
│       │   ├── CAM_FRONT_RIGHT
│       │   └── LIDAR_TOP -> ../../../../nuscenes/samples/LIDAR_TOP/
│       └── sweeps
│           └── LIDAR_TOP -> ../../../../nuscenes/sweeps/LIDAR_TOP/
├── dark
│   └── 3
│       ├── samples
│       │   ├── CAM_BACK
│       │   ├── CAM_BACK_LEFT
│       │   ├── CAM_BACK_RIGHT
│       │   ├── CAM_FRONT
│       │   ├── CAM_FRONT_LEFT
│       │   ├── CAM_FRONT_RIGHT
│       │   └── LIDAR_TOP -> ../../../../nuscenes/samples/LIDAR_TOP/
│       └── sweeps -> ../../../../nuscenes/sweeps
├── lidar_fog
│   └── 3
│       ├── samples
│       │   ├── CAM_BACK -> ../../../../nuscenes/samples/CAM_BACK
│       │   ├── CAM_BACK_LEFT -> ../../../../nuscenes/samples/CAM_BACK_LEFT
│       │   ├── CAM_BACK_RIGHT -> ../../../../nuscenes/samples/CAM_BACK_RIGHT
│       │   ├── CAM_FRONT -> ../../../../nuscenes/samples/CAM_FRONT
│       │   ├── CAM_FRONT_LEFT -> ../../../../nuscenes/samples/CAM_FRONT_LEFT
│       │   ├── CAM_FRONT_RIGHT -> ../../../../nuscenes/samples/CAM_FRONT_RIGHT
│       │   └── LIDAR_TOP
│       └── sweeps
│           └── LIDAR_TOP
├── lidar_motionblur
│   └── 3
│       ├── samples
│       │   ├── CAM_BACK -> ../../../../nuscenes/samples/CAM_BACK
│       │   ├── CAM_BACK_LEFT -> ../../../../nuscenes/samples/CAM_BACK_LEFT
│       │   ├── CAM_BACK_RIGHT -> ../../../../nuscenes/samples/CAM_BACK_RIGHT
│       │   ├── CAM_FRONT -> ../../../../nuscenes/samples/CAM_FRONT
│       │   ├── CAM_FRONT_LEFT -> ../../../../nuscenes/samples/CAM_FRONT_LEFT
│       │   ├── CAM_FRONT_RIGHT -> ../../../../nuscenes/samples/CAM_FRONT_RIGHT
│       │   └── LIDAR_TOP
│       └── sweeps
│           └── LIDAR_TOP

```
# Installation Steps

1. Clone the github repo. We will be mounting this repo into our docker container to ensure modifications in the Docker are saved onto local disk
2. Modify the docker container to build the correct mmcv for your compute version. Build the docker container: `docker build -t mmdet:latest .`
3. Run the docker container in silent mode: 
```
docker run -d --gpus all --shm-size=150gb \
    -v /data/jason/SWAN_Cam_Ready/SWAN/mmdetection3d:/workspace/mmdetection3d \
    -v /data/jason/nuScenes/:/workspace/mmdetection3d/data/nuscenes \
    -v /data/jason/multicorrupt/:/workspace/mmdetection3d/data/multicorrupt \
    --name mmdet_container \
    mmdet:latest \
    sleep infinity
```

Be sure to modify the paths to point to your cloned repo and also the data directories. Modify the RAM allocation as fit under --shm-size

4. Attach to the running docker container

5. After attaching, run `bash setup_env.sh` to override the existing Swin transformer libraries with our custom code.

6. python projects/BEVFusion/setup.py develop


# Training

Run the training script to train all the SWAN variants: `bash scripts/ECCV_train_corruptions.sh`

This script performs the four trianings:...

TO BE CONTINUED
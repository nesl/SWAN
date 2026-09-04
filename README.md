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

# Checkpoint Acquisition
1. Navigate to the following [link](https://ucla.box.com/s/5l3ui13yuxibdi7xwscyu5qjv92ovavx) to receive the multimodal LiDAR + Camera network (based on CMT) trained on nuScenes with a LayerDrop rate of 0.2
    - *cmt_swin_layerdrop*: This is the checkpoint of the unimiodal SWIN transformer trained with LayerDrop
    - *cmt_voxel_015_flatformer_layerdrop_group256_efficientvfe*: This is the LiDAR only model trained with LayerDrop
    - *cmt_voxel_015_flatformer_swin_both_pretrained_group_256_unfrozen_efficientvfe*: This is the multimodal network incorporating both checkpoints and retraining on nuScenes with LayerDrop
2. Only the multimodal checkpoint *cmt_voxel_015_flatformer_swin_both_pretrained_group_256_unfrozen_efficientvfe* is necessary, the other checkpoints can be used to reproduce the multimodal network training.
3. Download the multimodal network checkpoint and place it in under `mmdetection3d/work_dirs/cmt_voxel...`, creating the `work_dirs` directory if necessary.

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

6. Run `python projects/BEVFusion/setup.py develop` to setup the BEVFusion components


# Training
Run the training script to train all the SWAN variants: `bash scripts/ECCV_train_corruptions.sh`

This script performs the four trainings:

- First, we train the multimodal newtork on the multicorrupt nuScenes dataset with LayerDrop, as the previous checkpoint was only exposed to clean data
- Next, we train the universal SWAN QoI controller
- Afterwards, we load the previous checkpoint and train the SWAN SkipGate module
- Lastly, we train two stages of token pruning: soft pruning and hard pruning. Soft pruning replaces tokens with zero while hard pruning gets rid of the zero tokens. Doing it in two stages prevents the model from diverging during training.

`bash scripts/ECCV_train_baselines.sh` can be used to train the ADMN baseline for comparison

# Testing
`bash scripts/ECCV_test_corruptions.sh` will test all the SWAN models and all baselines (ADMN and Naive)




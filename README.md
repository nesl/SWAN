# Installation Steps

1. Clone the github repo. We will be mounting this repo into our docker container to ensure modifications in the Docker are saved onto local disk
2. Modify the docker container to build the correct mmcv for your compute version. Build the docker container: `docker build -t mmdet:latest`
3. Run the docker container in silent mode: 
```
docker run -d --gpus all --shm-size=150gb \
    -v /data/jason/Modern_UniM2AE/mmdetection3d:/workspace/mmdetection3d \
    -v /data/jason/nuScenes/:/workspace/mmdetection3d/data/nuscenes \
    --name mmdet_container \
    mmdet:latest \
    sleep infinity
```

Be sure to modify the paths to point to your cloned repo and also the data directory. Modify the RAM allocation as fit under --shm-size

4. Attach to the running docker container

# Pretraining Steps
1. Preprocess the data: `python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes`
2. Run the pretraining: `python3 tools/train.py projects/UniM2AE/configs/unim2ae_mmim.py`
Alternatively, with multiple GPUs: `CUDA_VISIBLE_DEVICES=X, X bash tools/dist_train.sh projects/UniM2AE/configs/unim2ae_mmim.py ${GPU_NUM}`

# Finetuning Steps

## Training the unimodal lidar model
Run `python tools/train.py projects/CMT/configs/cmt_voxel_015_flatformer.py` for single GPU or `CUDA_VISIBLE_DEVICES=2,3 bash tools/dist_train.sh projects/CMT/configs/cmt_voxel_015_flatformer.py 2` for multi-gpu

## Train the multimodal lidar + camera model
1. Update the `load_from` path in `projects/CMT/configs/cmt_voxel_015_flatformer_swin_from_lidar.py` to point to the lidar only flatformer model that we trained in the previous step
2. Run the multimodal fine-tuning: `python tools/train.py projects/CMT/configs/cmt_voxel_015_flatformer_swin_from_lidar.py` for single GPU or `CUDA_VISIBLE_DEVICES=2,3 bash tools/dist_train.sh projects/CMT/configs/cmt_voxel_015_flatformer_swin_from_lidar.py 2` for multi-gpu



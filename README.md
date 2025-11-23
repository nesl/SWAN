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

# Running Steps
1. Preprocess the data: `python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes`
2. Run the pretraining: `python3 tools/train.py projects/UniM2AE/configs/unim2ae_mmim.py`
Alternatively, with multiple GPUs: `CUDA_VISIBLE_DEVICES=X, X bash tools/dist_train.sh projects/UniM2AE/configs/unim2ae_mmim.py ${GPU_NUM}`

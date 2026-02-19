#!/bin/bash -l
#SBATCH --nodes=1 # Allocate *at least* 1 node to this job.
#SBATCH --ntasks=1 # Allocate *at most* 1 task for job steps in the job
#SBATCH --cpus-per-task=16 # Each task needs only one CPU
#SBATCH --mem=192gb # This particular job won't need much memory
#SBATCH --time=4-00:01:00  # 4 days and 1 minute
#SBATCH --job-name="trans_VL"
#SBATCH -p batch # You could pick other partitions for other jobs
#SBATCH --wait-all-nodes=1  # Run once all resources are available
#SBATCH --output=/data/HangQiu/proj/mlsys/DynMMOT_scripts/transfusion/logs/transfusion_VL.txt 

date
nvidia-smi


## Navigate to mmdetection3d root
cd /data/HangQiu/proj/mlsys/DynMMOT/mmdetection3d
pwd
## Set path
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=/data/HangQiu/proj/mlsys/DynMMOT/mmdetection3d:$PYTHONPATH
export PYTHONPATH=/data/HangQiu/proj/mlsys/DynMMOT/mmdetection3d/projects:$PYTHONPATH
export PYTHONPATH=/data/HangQiu/proj/mlsys/DynMMOT/mmdetection3d/projects/BEVFusion:$PYTHONPATH
export PYTHONPATH=/data/HangQiu/proj/mlsys/DynMMOT/mmdetection3d/projects/TransFusion:$PYTHONPATH
# GPU Settings
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

WORKDIR=/data/HangQiu/proj/mlsys/DynMMOT_scripts/transfusion/work_dir/transfusion_voxel
export PORT=$((29500 + RANDOM % 1000))

CFG=/data/HangQiu/proj/mlsys/DynMMOT/mmdetection3d/projects/TransFusion/configs/transfusion_nusc_voxel_L.py

bash tools/dist_train.sh \
    $CFG \
    $NUM_GPUS \
    --work-dir $WORKDIR \
    --cfg-options  \
        "train_dataloader.batch_size=16" \
        "train_dataloader.num_workers=32" \
        "val_dataloader.batch_size=4" \
        "val_dataloader.num_workers=8" \
        "default_hooks.checkpoint.out_dir=${WORKDIR}" \
        "data_root=/data/HangQiu/data/nuscenes/" \
        "db_sampler.data_root=/data/HangQiu/data/nuscenes/" \
        "db_sampler.info_path=/data/HangQiu/data/nuscenes/nuscenes_dbinfos_train.pkl" \
        "train_dataloader.dataset.dataset.pipeline.3.db_sampler.data_root=/data/HangQiu/data/nuscenes/" \
        "train_dataloader.dataset.dataset.pipeline.3.db_sampler.info_path=/data/HangQiu/data/nuscenes/nuscenes_dbinfos_train.pkl" \
        "train_dataloader.dataset.dataset.data_root=/data/HangQiu/data/nuscenes/" \
        "val_dataloader.dataset.data_root=/data/HangQiu/data/nuscenes/" \
        "test_dataloader.dataset.data_root=/data/HangQiu/data/nuscenes/" \
        "val_evaluator.data_root=/data/HangQiu/data/nuscenes/" \
        "val_evaluator.ann_file=/data/HangQiu/data/nuscenes/nuscenes_infos_val.pkl" \
        "test_evaluator.data_root=/data/HangQiu/data/nuscenes/" \
        "test_evaluator.ann_file=/data/HangQiu/data/nuscenes/nuscenes_infos_val.pkl" 
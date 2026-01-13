#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=multi_corrupt_img_convert
#SBATCH -p batch
#SBATCH --wait-all-nodes=1
#SBATCH --output=/home/csgrad/sjin075/logs/convert1.txt # logging per job and per host in the current directory. Both stdout and stderr are logged.

conda init
conda activate openmmdet


cd /data/HangQiu/proj/mlsys/DynMMOT/mmdetection3d/projects/MultiCorrupt/converter


# just run the commands directly if you don't want to use slurm
python img_converter.py \
--corruption dark \
--root_folder /data/HangQiu/data/nuscenes \
--dst_folder /data/HangQiu/data/multicorrupt/dark/1 \
--severity 1 \
--n_cpus 8

python lidar_converter.py \
--corruption beamsreducing \
--root_folder /data/HangQiu/data/nuscenes \
--dst_folder /data/HangQiu/data/multicorrupt/beamsreducing/1 \
--severity 1 \
--n_cpus 8



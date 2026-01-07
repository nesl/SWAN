FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel
ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1
# MMDetection3D
RUN pip install -U openmim
RUN mim install mmengine
RUN pip install ninja
# CHANGE TO YOUR GPU ARCHITECTURE
RUN MMCV_CUDA_ARGS='-gencode=arch=compute_89,code=sm_89' pip install mmcv==2.1.0 --no-binary mmcv
RUN mim install 'mmdet>=3.0.0'
RUN mim install "mmdet3d>=1.1.0"

# Packages
RUN apt-get update 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git sudo wget nano mlocate libgl1 libglib2.0-0 --no-install-recommends

WORKDIR /workspace

WORKDIR /workspace/mmdetection3d
RUN pip install timm
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
ENV PYTHONPATH="/workspace/mmdetection3d:${PYTHONPATH}"

RUN bash -c 'mv libraries_replace/swin.py /opt/conda/lib/python3.10/site-packages/mmdet/models/backbones/swin.py'

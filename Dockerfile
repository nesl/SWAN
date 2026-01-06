FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel
ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1
# MMDetection3D
RUN pip install -U openmim
RUN mim install mmengine
RUN pip install ninja
# CHANGE TO YOUR GPU ARCHITECTURE
RUN MMCV_CUDA_ARGS='-gencode=arch=compute_90,code=sm_90' pip install mmcv==2.1.0 --no-binary mmcv
RUN mim install 'mmdet>=3.0.0'
RUN mim install "mmdet3d>=1.1.0"

# Packages
RUN apt-get update 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git sudo wget nano mlocate libgl1 libglib2.0-0 --no-install-recommends

WORKDIR /workspace

WORKDIR /workspace/mmdetection3d
RUN pip install timm
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
ENV PYTHONPATH="/workspace/mmdetection3d:${PYTHONPATH}"

RUN bash -c 'mv libraries_replace/swin.py /opt/conda/lib/python3.10/site-packages/mmdet/models/backbones/swin.py'

#FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
#FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel


ENV TORCH_CUDA_ARCH_LIST="7.0 8.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which python))/../"

WORKDIR /pointr

RUN apt-get update && apt-get install -y --no-install-recommends git vim libegl1 libgl1 libgomp1 libosmesa6-dev && rm -rf /var/lib/apt/lists/*
# Configure Mesa EGL for headless rendering
ENV EGL_PLATFORM=surfaceless
RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
COPY extensions ./extensions
RUN pip install extensions/chamfer_dist extensions/cubic_feature_sampling extensions/emd extensions/gridding extensions/gridding_loss
COPY requirements.txt .
RUN pip install -r requirements.txt
# PointNet++
RUN pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
RUN pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

### QOL
RUN pip install wandb
# SSH for PyCharm remote debugging
ENV SSH_PASSWD "root:Ny~ln?FB,DxpsT=@7x6,p!o-ITHl[![*"
RUN apt-get update && apt-get install -y --no-install-recommends dialog  openssh-server \
    && echo "$SSH_PASSWD" | chpasswd
CMD service ssh restart && bash

## SKAIS
COPY skaislab skaislab
WORKDIR /pointr/skaislab
RUN apt-get install python3-dev

WORKDIR /pointr

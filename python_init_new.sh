#!/bin/bash

## Environment Variables
export MODULEPATH=/mnt/home/gkrawezik/modules/rocky8:$MODULEPATH

# Python Related Paths
export MPLCONFIGDIR=/mnt/home/acanatar/.config/matplotlib

# Torch Paths
export TORCH_HOME="~/.torch"

# Tensorflow Paths
export TFDS_CACHE_DIR="/mnt/ceph/users/acanatar/.cache/tensorflow_datasets"
export TFDS_DATA_DIR="/mnt/ceph/users/acanatar/tensorflow_datasets"
export TFHUB_CACHE_DIR="/mnt/ceph/users/acanatar/.cache/tfhub_modules"

# Huggingface Paths
export HF_HOME="/mnt/ceph/users/acanatar/huggingface"
export PROJECT_DIR="/mnt/ceph/users/acanatar/foundations"
export NEURAL_FOUNDATIONS_SRC="/mnt/home/acanatar/neural-foundations/src"
export ANACONDA_PATH=/mnt/home/acanatar/anaconda3/bin/activate
export TORCH_HOME=/mnt/home/acanatar/ceph/.torch

export EXPERIMENT_ROOT=/mnt/home/acanatar/ceph/manifold-experiments
export WANDB_API_KEY=101e8cacc42cb22ed02071571fb2017ddd6c5ade

## Activate the environment
# source /mnt/home/acanatar/miniconda3/bin/activate
source /mnt/home/acanatar/.bashrc
conda activate neural

## First purge the modules (If using Conda make sure python is not loaded)
module purge

## Load the modules (e.g. cudnn cuda gcc)
module load modules slurm cuda/12.1.1 cudnn/8.9.2.26 nccl gmp mpfr ffmpeg gcc llvm magma nvtop texlive jupyter-kernels openmpi/cuda-4.0.7 nvhpc fftw/nvhpc-3.3.10 intel-oneapi-mkl graphviz

## Print the loaded modules
module list

## Create a gpu node: -t <DD-HH:MM:SS>
srun -p gpu -C "v100" --gpus=1 --mem=32G -c 4 -t 1-23:59:00 \
    /mnt/home/acanatar/jupyter/neural-manifold-experiments/train.py \
    experiment_name=TESTING_LSTSQ_AGAIN_BRO \
    model=mmcr_two_stage/l1 \
    model.manifold_loss="radius" \
    optimizer=lars \
    learning_rate=1 \
    batch_size=2048 \
    dataset.frames_per_clip=6 \
    +model.projection_kwargs.features=[2048,512,128] \
    epochs=101 \
    seed=0

    # /mnt/home/acanatar/jupyter/neural-manifold-experiments/train.py \
    # dataset=moving_mnist \
    # experiment_name=test_new_forward \
    # model=mmcr_two_stage/l0 \
    # optimizer=lars \
    # learning_rate=5 \
    # batch_size=512 \
    # +model.projection_kwargs.features=[2048,512,128] \
    # model.encoder.type="convnext_tiny" \
    # manifold.return_nodes=['stages.0.blocks.0.conv_dw','stages.0.blocks.1.conv_dw','stages.0.blocks.2.conv_dw','stages.1.blocks.0.conv_dw','stages.1.blocks.1.conv_dw','stages.1.blocks.2.conv_dw','stages.2.blocks.0.conv_dw','stages.2.blocks.1.conv_dw','stages.2.blocks.2.conv_dw','stages.2.blocks.3.conv_dw','stages.2.blocks.4.conv_dw','stages.2.blocks.5.conv_dw','stages.2.blocks.6.conv_dw','stages.2.blocks.7.conv_dw','stages.2.blocks.8.conv_dw','stages.3.blocks.0.conv_dw','stages.3.blocks.1.conv_dw','stages.3.blocks.2.conv_dw']


    # /mnt/home/acanatar/jupyter/neural-manifold-experiments/train.py \
    # --multirun dataset=moving_mnist \
    # experiment_name=mmcr_training_frames_per_clip_convnext_projection_head \
    # model=mmcr_two_stage/l0,mmcr_two_stage/l1 \
    # optimizer=lars \
    # learning_rate=0.1 \
    # batch_size=512 \
    # dataset.frames_per_clip=2,3,4,5,6,7 \
    # model.encoder.type="convnext_tiny" \
    # +model.projection_kwargs.features=[3072,512,128] \
    # manifold.return_nodes=['stages.0.blocks.0.conv_dw','stages.0.blocks.1.conv_dw','stages.0.blocks.2.conv_dw','stages.1.blocks.0.conv_dw','stages.1.blocks.1.conv_dw','stages.1.blocks.2.conv_dw','stages.2.blocks.0.conv_dw','stages.2.blocks.1.conv_dw','stages.2.blocks.2.conv_dw','stages.2.blocks.3.conv_dw','stages.2.blocks.4.conv_dw','stages.2.blocks.5.conv_dw','stages.2.blocks.6.conv_dw','stages.2.blocks.7.conv_dw','stages.2.blocks.8.conv_dw','stages.3.blocks.0.conv_dw','stages.3.blocks.1.conv_dw','stages.3.blocks.2.conv_dw']








# --multirun 
# +model.freeze_until=conv1,layer1,layer2,layer3,layer4,autoregressive_head
# model.encoder.weights='${oc.env:EXPERIMENT_ROOT}/multirun/find_best_lr_mnist_linear/encoder.pth' \


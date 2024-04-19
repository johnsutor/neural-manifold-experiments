#!/bin/bash

## Environment Variables
export MODULEPATH=/mnt/home/gkrawezik/modules/rocky8:$MODULEPATH

# Python Related Paths
export MPLCONFIGDIR=/mnt/home/jsutor/.config/matplotlib

# Torch Paths
export TORCH_HOME="~/.torch"

# Tensorflow Paths
export TFDS_CACHE_DIR="/mnt/ceph/users/jsutor/.cache/tensorflow_datasets"
export TFDS_DATA_DIR="/mnt/ceph/users/jsutor/tensorflow_datasets"
export TFHUB_CACHE_DIR="/mnt/ceph/users/jsutor/.cache/tfhub_modules"

# Huggingface Paths
export HF_HOME="/mnt/ceph/users/jsutor/huggingface"
export PROJECT_DIR="/mnt/ceph/users/jsutor/foundations"
export NEURAL_FOUNDATIONS_SRC="/mnt/home/jsutor/neural-foundations/src"
export ANACONDA_PATH=/mnt/home/jsutor/anaconda3/bin/activate
export EXPERIMENT_ROOT=/mnt/home/jsutor/ceph/manifold-experiments
export TORCH_HOME=/mnt/home/jsutor/ceph/.torch

## Activate the environment
# source /mnt/home/acanatar/miniconda3/bin/activate
source /mnt/home/jsutor/.bashrc
conda activate neural

## First purge the modules (If using Conda make sure python is not loaded)
module purge

## Load the modules (e.g. cudnn cuda gcc)
module load modules slurm cuda/12.1.1 cudnn/8.9.2.26 nccl gmp mpfr ffmpeg gcc llvm magma nvtop texlive jupyter-kernels openmpi/cuda-4.0.7 nvhpc fftw/nvhpc-3.3.10 intel-oneapi-mkl graphviz

## Print the loaded modules
module list

## Create a gpu node: -t <DD-HH:MM:SS>
srun -p gpu -C "a100,ib" --gpus=1 --mem=1000G -c 8 -t 3-23:59:00 -N 1 \
    /mnt/home/jsutor/manifold-experiments/train.py \
    --multirun dataset=custom_moving_mnist \
    experiment_name=long_mmcr_training_frames_per_clip_resnet_projection_head_search_custom_moving_mnist \
    model=mmcr_two_stage/l0,mmcr_two_stage/l1 \
    optimizer=lars \
    learning_rate=5,1,0.5,0.1 \
    batch_size=512,1024,2048 \
    dataset.frames_per_clip=6 \
    +model.projection_kwargs.features=[2048,512,128] \
    epochs=101 \
    seed=0,42,1337,8675309,525600 
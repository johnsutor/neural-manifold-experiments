python /mnt/home/acanatar/jupyter/neural-manifold-experiments/train.py \
    --multirun dataset=custom_moving_mnist,moving_mnist \
    hydra/launcher=submitit_slurm \
    hydra.launcher.nodes=1 \
    hydra.launcher.mem_gb=64 \
    hydra.launcher.gpus_per_node=1 \
    hydra.launcher.constraint='"a100,ib"' \
    +hydra.launcher.gpus_per_task=1 \
    +hydra.launcher.cpus_per_task=4 \
    hydra.launcher.array_parallelism=8 \
    hydra.launcher.partition=gpu \
    hydra.launcher.setup="['export MODULEPATH=/mnt/home/gkrawezik/modules/rocky8:$MODULEPATH', 'export MPLCONFIGDIR=/mnt/home/acanatar/.config/matplotlib', 'export TORCH_HOME=\"~/.torch\"', 'export TFDS_CACHE_DIR=\"/mnt/ceph/users/acanatar/.cache/tensorflow_datasets\"', 'export TFDS_DATA_DIR=\"/mnt/ceph/users/acanatar/tensorflow_datasets\"', 'export TFHUB_CACHE_DIR=\"/mnt/ceph/users/acanatar/.cache/tfhub_modules\"', 'export HF_HOME=\"/mnt/ceph/users/acanatar/huggingface\"', 'export PROJECT_DIR=\"/mnt/ceph/users/acanatar/foundations\"', 'export NEURAL_FOUNDATIONS_SRC=\"/mnt/home/acanatar/neural-foundations/src\"', 'export ANACONDA_PATH=/mnt/home/acanatar/anaconda3/bin/activate', 'export EXPERIMENT_ROOT=/mnt/home/acanatar/ceph/manifold-experiments', 'export TORCH_HOME=/mnt/home/acanatar/ceph/.torch', \"source /mnt/home/acanatar/.bashrc\", \"conda activate neural\", \"module purge\", \"module load modules slurm cuda/12.1.1 cudnn/8.9.2.26 nccl gmp mpfr ffmpeg gcc llvm magma nvtop texlive jupyter-kernels openmpi/cuda-4.0.7 nvhpc fftw/nvhpc-3.3.10 intel-oneapi-mkl graphviz\", \"module list\"]" \
    experiment_name=mmcr_multiloss_250_epochs \
    model=mmcr_two_stage/l0,mmcr_two_stage/l1 \
    model.manifold_loss="capacity","radius","dimensionality" \
    optimizer=lars \
    learning_rate=1,0.5 \
    batch_size=512,1024,2048 \
    dataset.frames_per_clip=6 \
    +model.projection_kwargs.features=[2048,512,128] \
    epochs=251 \
    seed=0,42,1337,8675309,525600








# python /mnt/home/acanatar/jupyter/neural-manifold-experiments/train.py \
#     --multirun dataset=custom_moving_mnist,moving_mnist \
#     hydra/launcher=submitit_slurm \
#     hydra.launcher.nodes=1 \
#     hydra.launcher.mem_gb=64 \
#     hydra.launcher.gpus_per_node=1 \
#     hydra.launcher.constraint='"a100,ib"' \
#     +hydra.launcher.gpus_per_task=1 \
#     +hydra.launcher.cpus_per_task=4 \
#     hydra.launcher.array_parallelism=8 \
#     hydra.launcher.partition=gpu \
#     hydra.launcher.setup="['export MODULEPATH=/mnt/home/gkrawezik/modules/rocky8:$MODULEPATH', 'export MPLCONFIGDIR=/mnt/home/acanatar/.config/matplotlib', 'export TORCH_HOME=\"~/.torch\"', 'export TFDS_CACHE_DIR=\"/mnt/ceph/users/acanatar/.cache/tensorflow_datasets\"', 'export TFDS_DATA_DIR=\"/mnt/ceph/users/acanatar/tensorflow_datasets\"', 'export TFHUB_CACHE_DIR=\"/mnt/ceph/users/acanatar/.cache/tfhub_modules\"', 'export HF_HOME=\"/mnt/ceph/users/acanatar/huggingface\"', 'export PROJECT_DIR=\"/mnt/ceph/users/acanatar/foundations\"', 'export NEURAL_FOUNDATIONS_SRC=\"/mnt/home/acanatar/neural-foundations/src\"', 'export ANACONDA_PATH=/mnt/home/acanatar/anaconda3/bin/activate', 'export EXPERIMENT_ROOT=/mnt/home/acanatar/ceph/manifold-experiments', 'export TORCH_HOME=/mnt/home/acanatar/ceph/.torch', \"source /mnt/home/acanatar/.bashrc\", \"conda activate neural\", \"module purge\", \"module load modules slurm cuda/12.1.1 cudnn/8.9.2.26 nccl gmp mpfr ffmpeg gcc llvm magma nvtop texlive jupyter-kernels openmpi/cuda-4.0.7 nvhpc fftw/nvhpc-3.3.10 intel-oneapi-mkl graphviz\", \"module list\"]" \
#     experiment_name=mmcr_multiloss_250_epochs \
#     model=mmcr_two_stage/l0,mmcr_two_stage/l1 \
#     model.manifold_loss="capacity","radius","dimensionality" \
#     optimizer=lars \
#     learning_rate=1,0.5 \
#     batch_size=512,1024,2048 \
#     dataset.frames_per_clip=6 \
#     +model.projection_kwargs.features=[2048,512,128] \
#     epochs=251 \
#     seed=0,42,1337,8675309,525600

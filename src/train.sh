#!/bin/bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --job-name=beetles_ensemble
#SBATCH --output=beetles_go.out
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

time python train.py\
    --log_dir "$HOME"/beetles-logs/ \
    --vert_trim 30 \
    --n_fft 800 \
    --batch_size 512 \
    --data_path "$HOME/share/beetles-cnn/data/" \
    --in_channels 98 \
    --learning_rate 1e-3 \
    --gpus 1 \
    --num_nodes 1 \
    --epochs 300 \
    --num_workers 8 \
    --log \
    --mel

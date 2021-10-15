#!/bin/bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --job-name=beetles_ensemble
#SBATCH --output=beetles_go.out
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2

source ~/anaconda/bin/activate
conda activate beetles

time python train.py\
    --gpus 0 \
    --log_dir "$HOME"/beetles-logs/ \
    --vertical_trim 0 \
    --n_fft 800 \
    --batch_size 512 \
    --data_path "$HOME/share/beetles-cnn/data/" \
    --learning_rate 1e-3 \
    --begin_cutoff_idx 0 \
    --num_nodes 0 \
    --epochs 300 \
    --num_workers 36 \
    --log \
    --mel

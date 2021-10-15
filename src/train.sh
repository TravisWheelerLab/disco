#!/bin/bash

#SBATCH --partition=wheeler_lab_large_cpu
#SBATCH --job-name=beetles_ensemble
#SBATCH --output=beetles_go.out
#SBATCH --cpus-per-task=8

source ~/anaconda/bin/activate
conda activate beetles

time python train.py \
    --gpus 0 \
    --log_dir "$HOME"/beetles-logs/ \
    --vertical_trim 0 \
    --n_fft 800 \
    --batch_size 512 \
    --data_path "$HOME/share/beetles-cnn/data/" \
    --learning_rate 1e-3 \
    --begin_cutoff_idx 0 \
    --check_val_every_n_epoch 10 \
    --num_nodes 0 \
    --epochs 300 \
    --num_workers 32 \
    --log \
    --mel

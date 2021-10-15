#!/bin/bash

source ~/anaconda/bin/activate
conda activate beetles

python hparam_optimizer.py\
    --gpus 0 \
    --log_dir "$HOME"/beetles-logs/ \
    --vertical_trim 0 \
    --n_fft 800 \
    --batch_size 512 \
    --data_path "$HOME/share/beetles-cnn/data/" \
    --learning_rate 1e-3 \
    --num_nodes 0 \
    --epochs 10 \
    --check_val_every_n_epoch 5\
    --num_workers 8 \
    --log \
    --mel

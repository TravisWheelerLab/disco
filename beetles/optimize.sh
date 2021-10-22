#!/bin/bash

source ~/anaconda/bin/activate
conda activate beetles

python hparam_optimizer.py\
    --gpus 1 \
    --log_dir "$HOME"/beetles-logs/final_unet_hparam_tune/init_pow_6 \
    --vertical_trim 20 \
    --n_fft 650 \
    --batch_size 128 \
    --data_path "$HOME/data/beetles/extracted_data/" \
    --learning_rate 5e-3 \
    --num_nodes 1 \
    --epochs 300 \
    --check_val_every_n_epoch 10 \
    --num_workers 8 \
    --log \
    --mask_beginning_and_end \
    --begin_mask 28 \
    --end_mask 10 \
    --mel

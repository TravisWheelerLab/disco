#! /usr/bin/env bash
python hparam_optimizer.py\
    --gpus 1 \
    --log_dir "$HOME"/beetles-logs/ \
    --vert_trim 30 \
    --n_fft 800 \
    --batch_size 512 \
    --data_path "$HOME/share/beetles-cnn/data/" \
    --in_channels 98 \
    --learning_rate 1e-3 \
    --num_nodes 1 \
    --epochs 300 \
    --num_workers 8 \
    --log \
    --mel

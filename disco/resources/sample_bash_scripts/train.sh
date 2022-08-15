#!/bin/bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=ensemble-template
#SBATCH --output=/home/tc229954/beetles-logs/final_unet_ensemble/beetles-template.out

# source ~/anaconda/bin/activate

time disco train \
     --gpus 1 \
     --log_dir "$HOME"/disco-logs/final_unet/ \
     --vertical_trim 20 \
     --n_fft 1150 \
     --batch_size 128 \
     --data_path "$HOME/disco/disco/data/new_extracted_data_1150_nffts/" \
     --learning_rate 0.001 \
     --check_val_every_n_epoch 10 \
     --num_nodes 1 \
     --epochs 100 \
     --num_workers 0 \
     --begin_mask 28 \
     --end_mask 10 \
     --log \
     --mel \

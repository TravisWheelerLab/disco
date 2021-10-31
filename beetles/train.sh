#!/bin/bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=ensemble-template
#SBATCH --output=/home/tc229954/beetles-logs/final_unet_ensemble/beetles-template.out

source ~/anaconda/bin/activate
conda activate beetles

time beetles train \
     --gpus 1 \
     --log_dir "$HOME"/beetles-logs/final_unet_ensemble_with_correct_dimension/ \
     --vertical_trim 20 \
     --n_fft 1150 \
     --batch_size 128 \
     --data_path "$HOME/data/beetles/extracted_data" \
     --learning_rate 0.00040775 \
     --check_val_every_n_epoch 10 \
     --num_nodes 1 \
     --epochs 300 \
     --num_workers 32 \
     --mask_beginning_and_end \
     --begin_mask 28 \
     --end_mask 10 \
     --log \
     --mel \
     --bootstrap \

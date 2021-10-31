#!/usr/bin/env bash

python train_model.py \
  --batch-size 256\
  --shuffle\
  --learning_rate 1e-4\
   --epochs 325\
    --save_model\
     --mel\
      --log\
       --n_fft 800\
        --vert_trim 30\
         --bin_spects


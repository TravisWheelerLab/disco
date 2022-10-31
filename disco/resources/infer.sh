#!/usr/bin/env bash

time disco infer "./data/example.wav" \
     --tile_overlap 300 \
     --tile_size 1000 \
     --batch_size 32 \
     --input_channels 108 \
     --hop_length 200 \
     --vertical_trim 20 \
     --n_fft 1150 \
     --num_threads 4 \
     --noise_pct 0 \

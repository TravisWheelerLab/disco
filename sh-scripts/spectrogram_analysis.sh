#!/bin/bash

python spectrogram_analysis.py\
    --log_scale \
    --n_fft 800\
    --vert_trim 30\
    --save_fig \
    #     --mel_scale \
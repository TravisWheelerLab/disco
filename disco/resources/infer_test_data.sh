#!/bin/bash
for model_dir in /xdisk/twheeler/colligan/disco/disco/trained_models_beetle/*;
do
  bs=$(basename $model_dir)
  for snr in 10 20 40 80 160 320
  do
  data_dir=disco_noise_ablation/snr_$snr/test
  disco infer " " \
       --tile_overlap 300 \
       --tile_size 1000 \
       --batch_size 32 \
       --input_channels 108 \
       --hop_length 200 \
       --vertical_trim 20 \
       --n_fft 1150 \
       --num_threads 4 \
       --snr_ratio 0 \
       --saved_model_directory $model_dir \
       --viz_path "/xdisk/twheeler/colligan/disco_visualizations" \
       --map_unconfident \
       --low_confidence_iqr_threshold 0.05 \
       --model_extension "ckpt" \
       --accuracy_metrics \
       --accuracy_metrics_test_directory "/xdisk/twheeler/colligan/$data_dir" \
       --metrics_path "/xdisk/twheeler/colligan/disco_accuracy/snr_$snr"_$bs \
       --blackout_unconfident_in_viz
  done
done

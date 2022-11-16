#!/bin/bash
# line below iterates over all of the <ensemble_members/init method>
for model_dir in /xdisk/twheeler/colligan/disco/disco/trained_models_beetle/*;
do
  bs=$(basename $model_dir)
  # over all of the sn ratios tested
  for snr in 0
  do
  data_dir=disco_noise_ablation/snr_$snr/test
  mkdir -p "/xdisk/twheeler/colligan/disco_visualizations/hmm_ablation/$bs"
  disco infer " " \
       --tile_overlap 300 \
       --tile_size 1000 \
       --batch_size 32 \
       --input_channels 108 \
       --hop_length 200 \
       --vertical_trim 20 \
       --n_fft 1150 \
       --num_threads 4 \
       --snr 0 \
       --saved_model_directory $model_dir \
       --viz_path "/xdisk/twheeler/colligan/disco_visualizations/hmm_ablation/$bs" \
       --model_extension "ckpt" \
       --accuracy_metrics \
       --accuracy_metrics_test_directory "/xdisk/twheeler/colligan/$data_dir" \
       --metrics_path "/xdisk/twheeler/colligan/hmm_ablation/disco_accuracy/snr_$snr"_$bs
  done
done

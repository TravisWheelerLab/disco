#!/bin/bash
# line below iterates over all of the <ensemble_members/init method>
for model_dir in /xdisk/twheeler/colligan/disco/disco/trained_models_beetle/*;
do
  bs=$(basename $model_dir)
  # over all of the sn ratios tested
  for snr in 0 5 10 15 20 25 30 35 40 80 160 320
  do
  data_dir=/xdisk/twheeler/colligan/disco_noise_ablation/snr_$snr/test
  out_dir=/xdisk/twheeler/colligan/beetles_analysis/evaluated_test_files/snr_"$snr"_"$bs"/
  echo $out_dir
  if [[ ! -d $out_dir ]];
  then
    /opt/ohpc/pub/apps/python/3.8.2/bin/python3.8 disco/evaluate_beetles_test_files.py with test_path=$data_dir metrics_path=$out_dir saved_model_directory=$model_dir
  fi
  done
done

#!/bin/bash
# line below iterates over all of the <ensemble_members/init method>
# for model_dir in /Users/wheelerlab/trained_models_beetle/*;
for model_dir in /xdisk/twheeler/colligan/trained_models_beetle/*
do
  for sn_ratio in 0 5 10 15 20 25 30 35 40 45 50 100 150 200
  do
  bs=$(basename $model_dir)
  for data_file in 180101_0133S12.wav 180101_0183S34D06.wav trial40_M57_F29_070220.wav;
  do
    # root="/Users/wheelerlab/beetles_testing/unannotated_files_12_12_2022/audio_2020/"
    root="/xdisk/twheeler/colligan/ground_truth/"
    out_root="/xdisk/twheeler/colligan/ground_truth/snr_ablation/"
    data_file=$root/$data_file
    disco infer with wav_file=$data_file dataloader_args.snr=$sn_ratio saved_model_directory=$model_dir output_directory=$out_root/"$sn_ratio"_"$bs"
  done
  done
done
for i in {1..10}
do
x=$RANDOM
for data_file in 180101_0133S12.wav 180101_0183S34D06.wav trial40_M57_F29_070220.wav;
do
  disco infer with wav_file=$data_file output_directory=test_single_model/$x
done
done

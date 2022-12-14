#!/bin/bash
# line below iterates over all of the <ensemble_members/init method>
for model_dir in /Users/wheelerlab/trained_models_beetle/*;
do
  bs=$(basename $model_dir)
  for data_file in 180101_0133S12.wav 180101_0183S34D06.wav trial40_M57_F29_070220.wav;
  do
    root="/Users/wheelerlab/beetles_testing/unannotated_files_12_12_2022/audio_2020/"
    data_file=$root/$data_file
    disco infer with wav_file=$data_file saved_model_directory=$model_dir output_directory=$root/$bs
  done
done

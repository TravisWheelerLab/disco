#!/usr/bin/env bash
#SBATCH --partition=wheeler_lab_gpu
#SBATCH --job-name=beetles_ensemble
#SBATCH --output=beetles_label.out
#SBATCH --gres=gpu:1

data_path=/home/tc229954/data/beetles/all_recordings/
# i know this could be one line but it would require annoying xargs stuff
find $data_path \( -name "*.wav" -o -name "*.WAV" \) | while read file;
do
  dirname=$(echo $file | awk -F\/ '{print $(NF-1)}')
  x=$(basename "$file")
  y="${x%%.*}"
  x="${x%%.*}"
  echo "$x"
  python infer.py --saved_model_directory /home/tc229954/beetles-logs/final_unet_ensemble/models/ --wav_file $file --output_csv_path /home/tc229954/data/beetles/ensemble_results/predictions/$dirname/"$x"_predictions.csv --debug /home/tc229954/data/beetles/ensemble_results/debug/$dirname/$x --vertical_trim 20 --n_fft 1150 --input_channels 108
done

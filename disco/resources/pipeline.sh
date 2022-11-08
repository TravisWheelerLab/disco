#!/usr/bin/env bash

data_dir="/home/ki117611/extracted_data_1150_nffts"

echo "Running extraction."
# shellcheck disable=SC2162
find /mnt/beegfs/tc229954/beetles_data/ -name "*.csv" | grep -v predictions | grep -v debug | while read file;
do
  wav="${file%%.*}".wav
  disco extract --mel_scale "$file" "$wav" $data_dir
done

echo "PLEASE VISUALLY CHECK ALL EXTRACTED DATA."
echo "Running shuffle."
disco shuffle "$data_dir"
# shellcheck disable=SC2012
train_num=$(ls "$data_dir"/train | wc -l)
# shellcheck disable=SC2012
test_num=$(ls "$data_dir"/test | wc -l)
# shellcheck disable=SC2012
validation_num=$(ls "$data_dir"/validation | wc -l)
echo "num. train files: $train_num"
echo "num. test files: $test_num"
echo "num. validation files: $validation_num"

echo "Running train."

for method in random_init bootstrap;
do
for members in 2 10 30;
  do
  # shellcheck disable=SC2004
  for ((i = 0 ; i < $members ; i++ ))
    do
      echo "training model $i with method $method $members"
      bash disco/train.sh "trained_models_beetle_sept17/ensemble_""$members"_"$method" $method "$data_dir"
    done
  done
done

for method in random_init bootstrap;
do
  for members in 2 10 30;
    do
        bash disco/infer.sh "$data_dir"/test "trained_models_beetle/ensemble_""$members"_"$method"
    done
done
# THEN COMPUTE ACCURACY.
# PRINTS OUT the stuff
python3 disco/accuracy_metrics.py 2 "random_init"
python3 disco/accuracy_metrics.py 2 "bootstrap"

python3 disco/accuracy_metrics.py 10 "random_init"
python3 disco/accuracy_metrics.py 10 "bootstrap"

python3 disco/accuracy_metrics.py 30 "random_init"
python3 disco/accuracy_metrics.py 30 "bootstrap"

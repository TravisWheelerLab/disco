#!/bin/bash
# line below iterates over all of the <ensemble_members/init method>
csv_root="/xdisk/twheeler/colligan/disco_accuracy_csvs/"
data_root="/xdisk/twheeler/colligan/disco_accuracy/"
for method in "bootstrap" "random_init"
do
for ensemble_member in 2 10 30
do
  # over all of the sn ratios tested
  for snr in 0 10 20 40 80 160 320
  do
    infer_data_root=$data_root/"snr"_"$snr"_"ensemble_$ensemble_member"_"$method"
    out_path=$csv_root/"snr"_"$snr"_"ensemble_$ensemble_member"_"$method"
    python disco/accuracy_metrics.py $infer_data_root $out_path $ensemble_member $method
  done
done
done

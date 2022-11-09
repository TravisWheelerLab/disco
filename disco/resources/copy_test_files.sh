#!/bin/bash

cd /xdisk/twheeler/colligan/disco_noise_ablation
for dir in *;
do
mkdir $dir/test
echo $dir
cat /home/u4/colligan/disco/disco/resources/disco_test_files.txt | while read line;
do
  cp $dir/$line $dir/test/$line
done
done

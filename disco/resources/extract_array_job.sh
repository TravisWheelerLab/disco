#!/bin/bash

# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=extract
### Optional. Set the output filename.
### SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=outfiles/extract-%x-%j.out
### REQUIRED. Specify the PI group for this job (twheeler).
#SBATCH --account=twheeler
### REQUIRED. Set the partition for your job. Four partitions are available in
### the arizona cluster system: standard (uses group's monthly allocation of
### resources), windfall (does NOT use up your monthly quota, but jobs run in
### this partition can be interrupted), high_priority (requires purchasing
### compute resources), and qualified. You'll probably want to use one of
### <standard,windfall>
#SBATCH --partition=standard
### REQUIRED. Set the number of cores that will be used for this job.
#SBATCH --ntasks=12
### REQUIRED. Set the number of nodes
#SBATCH --nodes=1
### REQUIRED. Set the memory required for this job.
#SBATCH --mem-per-cpu=5gb
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=10:01:00
### any other slurm options are supported, but not required.
#SBATCH --array=[1-151]%100

## extract bits of data per labeled file
module load python/3.8
source ~/miniconda3/bin/activate
conda activate faiss
fname=/home/u4/colligan/disco/disco/resources/extraction_files.txt
test_files=/home/u4/colligan/disco/disco/resources/disco_test_files.txt
csv_file=$(sed -n "$SLURM_ARRAY_TASK_ID"p $fname)
wav="${csv_file%%.*}".wav
for snr in 0 5 10 15 20 25 30 35 40
do
  data_dir="/xdisk/twheeler/colligan/disco_noise_ablation/snr_$snr"
  disco extract --mel_scale "$csv_file" "$wav" "$data_dir" --snr $snr
  test_dir=$data_dir/test
done

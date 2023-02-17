DISCO annotates sound files with neural networks. 
See wiki links on the right hand side for more information 
about what this tool can do. You'll probably want to know how to 
[install](https://github.com/TravisWheelerLab/disco/wiki/Installation) and then [predict new .wav files](https://github.com/TravisWheelerLab/disco/wiki/Inference-using-pretrained-models) before anything else.


# Introduction
DISCO is designed to be an end-to-end machine-learning based sound annotation tool. As such, it contains command for the
most common steps of a machine learning pipeline. 
In order (and with corresponding CLI commands):

```
disco label with wav_file=/your/file/here output_csv_path=/your/out/file/here
``` 

Labeling data is often the first part of a machine learning pipeline as 
soon as the problem has been defined. The `disco label` command will start a `matplotlib`-based labeling app. See 
[the labeling guide](https://github.com/TravisWheelerLab/disco/wiki/Labeling-and-extracting-data.md) for more detailed instructions.

After labeling one or many sound files, you'll be left with `.csv.` files that contain the location, class, and duration of 
given sounds. Extract labeled regions of `.wav` files with
```
disco extract with wav_file=/your/wav/file csv_file=/your/csv/file output_data_path=/your/out/path/here
```
Importantly, `disco extract` will stitch together contiguous labeled regions (for example, a contiguous region with labels
A-B-A) and save them as one example. Files are saved with the python `pickle` module, and have the suffix `.pkl`.

After extracting data from `.wav` files to a location on disk, you'll want to create a train/test/validation split.
This is done via 
```shell
disco shuffle with data_directory=/where/you/saved/the/extracted/data
```
By default, 80% of the labels are train and the remaining 20% are allocated evenly across the validation and test set.
`disco` saves the split data in three subdirectories of the directory passed in to the `shuffle` command:

```
. <- directory passed into disco shuffle
├── test
├── train
└── validation
```
Two more commands make up the entirety of a machine learning pipeline:
`disco train` and `disco infer`. `disco train` is complex and will require source code modification for custom models, 
loss functions, or data sources.
See [this wiki article](https://github.com/TravisWheelerLab/disco/wiki/Training-new-models.md) for more information on training.

DISCO will predict a novel `.wav` file through
```shell
disco infer with wav_file=/your/file/here/
```
By default, `disco` will create a new directory for the results of its analysis. Currently, `disco` uses data and models 
from Japanese rhinocerous beetles and will download pretrained models the first time `disco infer` is invoked.
































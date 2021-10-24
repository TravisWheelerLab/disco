*Warning:* This is still under development and not very extensible.
Currently, only three labels are supported: A, B, and background. 

Use this tool to create labels to train a custom model on any .wav data.
After labeling is done, extract labels for training with `beetles extract`.

### Using:
```
beetles label --wav_file <> --output_csv_path <>
```
Try it with 
```
beetles label --wav_file resources/example.wav --output_csv_path test_labels.csv
```

## Extracting
Spectrograms often aren't fully annotated so saving only the labeled bits of them is necessary
for training NNs quickly. `beetles extract` will do the job, but currently it expects a rigid directory structure.
```
beetles extract --data_dir data_dir --output_data_path <> <OPTS>
```
This assumes a directory structure like below, where the .csv files contain
labels for the .wav file in the same directory.
```
data_dir
   ├── M12F31_8_24
   │   ├── 2_M12F31_8_24.WAV
   │   └── 2_M12F31_8_24.csv
   ├── M14F15_8_7
   │   ├── 1_M14F15_8_7.WAV
   │   └── 1_M14F15_8_7.csv
```
`beetles extract` can accept a few other arguments, like the number of ffts to use when calculating the spectrogram and whether or not to use a MelSpectrogram.



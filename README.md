# beetles-cnn
Pipeline for labeling, training, and classifying beetle chirps in .wav files.
This software is general enough to label and classify any .wav data.
# Installing
```
pip install beetles
```
# Predicting and visualizing data with pretrained models
```bash
beetles {infer, train, label, extract, viz} <OPTS>
```
See more information about any of the actions with `beetles <action> --help`
### beetles infer

Ingests a .wav file and predicts it with the ensemble of saved models. You can specify a  
`--saved_model_directory` but by default infer.py will download all required models to `$HOME/.cache/beetles/`.
Predictions are produced in .csv and saved at --output_csv_path in the same format as Raven,
shown below.
```
Selection,View,Channel,Begin Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Sound_Type
1,0,0,0.0,2.120833333333333,0,0,BACKGROUND
2,0,0,2.1666666666666665,5.079166666666667,0,0,BACKGROUND
```
Contains all 0s for data that is usually calculated by Raven.
### saving debugging/visualization data
```bash
beetles infer <opts> --debug PATH
```
Save the five necessary files required to run `beetles viz --debug_data_path` in ```$PWD/debug/```.
Will save the raw spectrogram, predictions post-hmm, the median prediction of the ensemble 
per class, uncertainty of the ensemble, and the .csv file containing the start and end of each
sound type.
Displays a matplotlib plot with a slider to ease visual inspection of the whole spectrogram. This utility is useful for
when you've implemented a new heuristic or changed the hmm and want to see what the large-scale impacts of your changes
are. Slide along the x-axis by clicking and dragging the slider on the bottom.

# Example
resources/example.wav contains a small wav file for testing the software. Run this command
to predict the wav file in `resources/`
```
beetles infer --wav_file ../resources/example.wav --debug <your_path_here>
```
then 
```
python interactive_plot.py --debug_data_path <your_path_here>
```
# Labeling data and training new models

This will bring up a simple matplotlib gui (controlled by mouse and keyboard)
that is sufficient for labeling sound types.
```
beetles label --wav_file <your_file_here> --output_csv_path output.csv
```
Since there are various hyperparameters associated with spectrogram calculation, the labels
are saved as a .csv. Extract the labeled bits of spectrogram with
```
beetles extract --data_dir <> --output_data_path <> --other_opts
```
This assumes a directory structure like below, where the .csv files contain
labels for the .wav file in the same directory.
```bash
├── M12F31_8_24
│   ├── 2_M12F31_8_24.WAV
│   └── 2_M12F31_8_24.csv
├── M14F15_8_7
│   ├── 1_M14F15_8_7.WAV
│   └── 1_M14F15_8_7.csv
```
TODO: Change this!
### training a new model
```bash
beetles train args
```
There are many hyperparameters associated with training a model. TODO: add default 
hparams.


# Contributing
Fork this repo and create a new branch.
Run `python setup.py develop` to see changes as you edit the source code.

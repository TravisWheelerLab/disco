You can perform inference on a .wav file with `beetles infer`.

```
beetles infer --wav_file <example.wav> <OPTS>
```

*Note:* The inference script evaluates the spectrogram with 10 different models
(same architecture, different weights) to estimate predictive uncertainty. GPUs
are recommended. 

The infer command can save predictions and visualization/debugging data.

Save predictions with `--output_csv_path <my_results.csv>` 

Save data for visualizing predictions in matplotlib with 
`--debug <directory_name_here>`. More information on how to use the --debug flag
is [here](Visualizing-data.md).

Results are saved as follows for easy visualization in Raven.
```
Selection,View,Channel,Begin Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Sound_Type
1,0,0,0.0,2.120833333333333,0,0,BACKGROUND
2,0,0,2.1666666666666665,5.079166666666667,0,0,BACKGROUND
```

# Example
`resources/example.wav` contains a small wav file for running the software. Run
this command to predict the wav file in `resources/`. You can either clone to
repo for access to resources/ or download the files manually.
```
beetles infer --wav_file resources/example.wav --debug <your_path_here>
```
then, to visualize:
```
beetles viz --debug_data_path <your_path_here>
```

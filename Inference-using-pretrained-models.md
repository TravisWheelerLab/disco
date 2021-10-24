You can perform inference on a .wav file with `beetles infer`.

```
beetles infer --wav_file <example.wav> <OPTS>
```

*Note:* The inference script evaluates the spectrogram with 10 different models
(same architecture, different weights) to estimate predictive uncertainty. GPUs
are recommended. 

The infer command can save predictions and debugging data.

Save predictions with `--output_csv_path <my_results.csv>` 

Save data for visualizing predictions in matplotlib with 
`--debug <directory_name_here>`. More information on how to use the --debug flag
is [here](Visualizing-data.md). . 


```
Selection,View,Channel,Begin Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Sound_Type
1,0,0,0.0,2.120833333333333,0,0,BACKGROUND
2,0,0,2.1666666666666665,5.079166666666667,0,0,BACKGROUND
```

### saving debugging/visualization data
```bash
beetles infer <opts> --debug PATH
```
Save the five necessary files required to run `beetles viz --debug_data_path` in 
`$PWD/debug/`


Will save the raw spectrogram, predictions post-hmm, the median prediction of
the ensemble per class, uncertainty of the ensemble, and the .csv file
containing the start and end of each sound type.  Displays a matplotlib plot
with a slider to ease visual inspection of the whole spectrogram. This utility
is useful for when you've implemented a new heuristic or changed the hmm and
want to see what the large-scale impacts of your changes are. Slide along the
x-axis by clicking and dragging the slider on the bottom.

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

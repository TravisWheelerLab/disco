You can perform inference on a .wav file with `beetles infer`. Unless you specify a `--saved_model_directory` models will be downloaded from an s3 bucket to `$HOME/.cache/beetles/`.

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
is [here](https://github.com/TravisWheelerLab/beetles-cnn/wiki/Visualizing-data)

Results are saved as comma-delimited files with the same columns as
[Raven](https://ravensoundsoftware.com/knowledge-base/selection-labels/).
The models are only set up to predict three Sound_Types: A, B, and Background.

```
Selection,View,Channel,Begin Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Sound_Type
1,0,0,0.0,2.120833333333333,0,0,BACKGROUND
2,0,0,2.1666666666666665,5.079166666666667,0,0,BACKGROUND
```

## Example
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

### How inference works

10 neural networks were trained on subsets of the training dataset (this
technique is often known as bagging) with the same sets of hyperparameters.
Hyperparameters were tuned by training models on the entire training set without
subsampling and recording the best performance on the test set.

The 10 models are applied to the same data at inference time to get a range of
opinions about what class a given timepoint in a .wav file belongs to. The
median of these predictions is considered the prediction of the ensemble, and
the inter-quartile range of the predictions the uncertainty. Predictions that
are above a uncertainty threshold are then thrown out and a hidden markov model
is applied to the thresholded predictions to enforce continuity.

Predictions post-hmm are then processed and saved to the final output csv.

TODO: add cli arg to inference routine to let the user specify the uncertainty
threshold.

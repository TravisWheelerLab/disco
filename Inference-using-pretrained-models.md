You can perform inference on a .wav file with `disco infer`.

```
disco infer with wav_file=</your/file/here/example.wav> 
```

*Note:* The inference script evaluates the spectrogram with 10 different models
(same architecture, different weights) to estimate predictive uncertainty. GPUs
are recommended.

Results are saved as comma-delimited files with the same columns as
[Raven](https://ravensoundsoftware.com/knowledge-base/selection-labels/).
The models right are only set up to predict three Sound_Types: A, B, and Background, and output looks
something like this:
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
disco infer with wav_file=resources/example.wav
```

### How inference works

10 neural networks were trained on subsets of the training dataset with the same sets of hyperparameters.
Hyperparameters were tuned by training models on the entire training set without
subsampling and recording the best performance on the test set.

The 10 models are applied to the same data at inference time to get a range of
opinions about what class a given timepoint in a .wav file belongs to. The
median of these predictions is considered the prediction of the ensemble, and
the inter-quartile range of the predictions the uncertainty. Predictions that
are above a uncertainty threshold are then thrown out and a hidden markov model
is applied to the thresholded predictions to enforce continuity.

Predictions post-hmm are then processed and saved to the final output csv.

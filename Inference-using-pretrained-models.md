You can perform inference on a .wav file with `beetles infer`. Unless you specify a `--saved_model_directory` models will be downloaded from an s3 bucket to `$HOME/.cache/beetles/`. The hyperparameters that were used to train the models in the s3 bucket are defaults in `beetles infer`.

```
beetles infer --wav_file <example.wav> <OPTS>
```

*Note:* The inference script evaluates the spectrogram with 10 different models
(same architecture, different weights) to estimate predictive uncertainty. GPUs
are recommended. 

Save predictions with `--output_csv_path <my_results.csv>` 

Save data for visualizing predictions in matplotlib with 
`--debug <directory_name_here>`. More information on how to use the --debug flag
is [here](https://github.com/TravisWheelerLab/beetles-cnn/wiki/Visualizing-data).

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
beetles infer --wav_file resources/example.wav --debug <your_path_here> --output_csv_path example_preds.csv
```
then, to visualize:
```
beetles viz --debug_data_path <your_path_here>
```
## Running inference/visualization on the cluster
Log in:
```
ssh -Y <your_username>@<cluster_address> # -Y means enable trusted Xwindow forwarding
```
Create a conda env:
```
conda create -n beetles python=3.8
conda activate beetles 
pip install beetles
```
Ask for an interactive session with a GPU:
```
srun --partition <your_gpu_partition_name> --gres=gpu:1 --pty bash
```
Once you're allocated resources, run the inference script with
```
beetles infer --wav_file <your file> --output_csv_path <your desired path> --debug <where to save the debugging data for viz.>
```
Then, type `exit` or Ctrl-D to quit the interactive session.
Use the login node of the cluster to run the visualization script. The -Y option on ssh allows us to use X11 forwarding.
```
beetles viz --debug_data_path <same path as above>
```

Alternatively, you can wrap the inference routine in a python script and use sbatch to submit the inference job.
Here's an example script:
```python
from beetles.infer import run_inference

run_inference(
    wav_file="/where/you/saved/the/wav/file",
    output_csv_path="predictions.csv",
    debug='/save/debug/data/here'
)
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

# Reproducing the  results in the preprint

First, you'll need the test split of the dataset to evaluate on.
The analysis can be split up into multiple steps:
1. Use ensembles of models to `infer` (really, predict) all of the test files.
   1. if you want to test how the models do with the addition of noise, you have to _add_ the noise in to the test .wav files and re-extract the data. The steps for this are in the bottom of the document.
2. The test dataset is a collection of discontinuous slices of spectrogram that are labeled as `A`, `B`, or `background`. When you use `evaluate_beetles_test_files.py`, the slices of spectrograms are concatenated into one long array and saved to disk. The main point here is that you need to run `python evaluate_beetles_test_files.py`. Here's an example: <br>`python disco/evaluate_beetles_test_files.py with test_path=my/path/ model_name=UNet1D metrics_path=/where/i/want/to/save/the/evaluated/data`.</br> This will take all of the `.pkl` files in `my/path/`, evaluate them with `UNet1D`, and save the results to `metrics_path`. If you have trained a unique model, specify the name in place of `UNet1D` and additionaly specify a `saved_model_directory` where your model checkpoints are saved.
3. Now, use `python disco/accuracy_metrics.py /metrics/path/ /where/to/save/the/csv` to actually compute accuracy metrics like precision, recall, etc. There are also a couple more arguments specifying the number of ensemble members and ensemble type, but you don't really need to worry about these.
4.

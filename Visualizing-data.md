`beetles viz --debug_data_path <output of beetles infer>`

`--debug_data_path` needs to contain five files 
(produced by `beetles infer --debug <viz_path`): classifications.csv,
raw_spectrogram.pkl, hmm_predictions.pkl, median_predictions.pkl, iqrs.pkl. 


This command displays an matplotlib window with a slider to allow interactive
visualization of ensemble predictions.

[Image shown by beetles viz.](https://github.com/TravisWheelerLab/beetles-cnn/blob/infer/resources/example_inference_image.png)
The app displays the raw spectrogram as the top image, along with annotations
for beginning and ends of different chirps. Background labels aren't shown.

The second image shows the median predictions as probabilities, the
uncertainties associated with each prediction (labeled iqrs, for inter-quartile
ranges), the prediction of the ensemble as a discrete number (labeled median
argmax), and the predictions post-hmm. The predictions post-hmm are used to
create the vertical lines in the spectrogram plot.

### What do colors mean in `beetles viz`?

The neural networks only predict three classes (A, B, and background) and as
such we map each prediction to an RGB code. This means a prediction of A (output
of the ensemble in this case is [1, 0, 0]) gets visualized as red. Accordingly,
bright red, green, or blue in the first row means the ensemble was very
confident that the chirp was A, B, or background, respectively.

The visualization of the uncertainty (iqr) is the same. Black means uncertainty
was close to 0 for all classes (an output of [0, 0, 0]), and brighter colors
mean more uncertainty. For example, cyan is encoded in RGB as [0, 1, 1]. This
color means that the ensemble had high uncertainty between classes B and
background (indices 1 and 2). In the image above, cyan often appears at the
transition between B chirps and background, meaning that different members of
the ensemble learned slightly different definitions of background and B chirp.

When trying to interpret a new color (yellow in the uncertainty row, for
example), just find the RGB vector that describes it and you'll have access to
between which classes the uncertainty lies.


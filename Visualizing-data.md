`beetles viz --debug_data_path <output of beetles infer>`

`--debug_data_path` needs to contain five files 
(produced by `beetles infer --debug <viz_path`): classifications.csv,
raw_spectrogram.pkl, hmm_predictions.pkl, median_predictions.pkl, iqrs.pkl. 


This command displays an matplotlib window with a slider to allow interactive
visualization of ensemble predictions.

[Image produced by beetles infer with --debug](resources/example_inference_image.png)
The app shows the raw spectrogram on the top, the median predictions of the
ensemble as probabilities, the uncertainties, 


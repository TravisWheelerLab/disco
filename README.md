[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

beetles-cnn classifies chirps with machine learning. 

Quickstart:
Install from source. Clone this repo and run
```
flit install -s
```

Learn how to use the tools provided in this package [here](https://github.com/TravisWheelerLab/beetles-cnn/wiki).

Python example of inference:
```python
from beetles.infer import run_inference
run_inference(wav_file='/path/to/my/wav/file',
              output_csv_path='/where/i/want/to/store/my/predictions')
```

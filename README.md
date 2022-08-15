[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

# DISCO Implements Sound Classification Obediently
This tool annotates sound files using neural networks. It uses a 1D architecture based on U-Net with additional 
post-processing heuristics including a Hidden Markov Model. 

DISCO is ideal for long streams of sound that need to be classified over time, producing output fully compatible with 
The Cornell Lab of Ornithology's sound tool [RAVEN](https://ravensoundsoftware.com/). Work is currently underway to 
annotate short samples of data with a single label. DISCO began jointly with the University of Montana's [Emlen Lab](https://hs.umt.edu/dbs/labs/emlen/) 
as an annotator for Japanese and Taiwanese Rhinoceros Beetle courtship songs, but it now generalizes to any kind of 
recording.

## Quickstart
Install requires python version 3.8.1.
Clone this repo and run:
```
flit install -s
pip install -r requirements.txt
```
## Tutorial
Visit this [YouTube link](https://www.youtube.com/watch?v=g0rIpVOpXZ4) for a thorough setup tutorial.

Learn more about how to use the tools provided in this package in the [wiki](https://github.com/TravisWheelerLab/disco/wiki).
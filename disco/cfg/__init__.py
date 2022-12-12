import os

from .extract_config import *
from .infer_config import *
from .label_config import *
from .shuffle_config import *
from .train_config import *
from .viz_config import *

aws_download_link = (
    "https://disco-models.s3.us-west-1.amazonaws.com/random_init_model_{}.ckpt"
)
default_model_directory = os.path.join(os.path.expanduser("~"), ".cache", "disco")
mask_flag = -1
name_to_class_code = {"A": 0, "B": 1, "BACKGROUND": 2, "X": 2}
class_code_to_name = {0: "A", 1: "B", 2: "BACKGROUND"}

name_to_rgb_code = {"A": "#B65B47", "B": "#A36DE9", "BACKGROUND": "#AAAAAA"}

hmm_transition_probabilities = [
    [0.995, 0.00000, 0.005],
    [0.0000, 0.995, 0.005],
    [0.00001, 0.00049, 0.9995],
]

hmm_start_probabilities = [0, 0, 1]
hmm_emission_probabilities = [
    {0: 0.995, 1: 0.00005, 2: 0.00495},
    {0: 0.1, 1: 0.88, 2: 0.020},
    {0: 0.05, 1: 0.05, 2: 0.9},
]

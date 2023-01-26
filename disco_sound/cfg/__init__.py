import os

from sacred import Experiment

aws_download_link = (
    "https://disco-models.s3.us-west-1.amazonaws.com/random_init_model_{}.ckpt"
)
default_model_directory = os.path.join(os.path.expanduser("~"), ".cache", "disco_sound")
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

viz_experiment = Experiment()
train_experiment = Experiment()
extract_experiment = Experiment()
infer_experiment = Experiment()
label_experiment = Experiment()
shuffle_experiment = Experiment()


@train_experiment.config
def _inject_semi_permanent():
    """
    Semi-permanent arguments used for the beetles dataset.
    """
    name_to_class_code = {"A": 0, "B": 1, "BACKGROUND": 2, "X": 2}
    class_code_to_name = {0: "A", 1: "B", 2: "BACKGROUND"}

    name_to_rgb_code = {"A": "#B65B47", "B": "#A36DE9", "BACKGROUND": "#AAAAAA"}
    visualization_columns = 600

    excluded_classes = ("Y", "C")

    mask_flag = -1
    default_spectrogram_num_rows = 128


@label_experiment.config
def _label_semi_permanent():
    visualization_n_fft = 1150
    vertical_cut = 20
    key_to_label = {"z": "A", "b": "B", "x": "BACKGROUND"}


@infer_experiment.config
def _infer_semi_permanent():
    saved_model_directory = os.path.join(
        os.path.expanduser("~"), ".cache", "disco_sound"
    )


@extract_experiment.config
def _extract_semi_permanent():
    seed = 0
    no_mel_scale = False
    n_fft = 1150
    overwrite = False
    snr = 0
    add_beeps = False
    extract_context = False
    class_code_to_name = {0: "A", 1: "B", 2: "BACKGROUND"}
    name_to_class_code = {"A": 0, "B": 1, "BACKGROUND": 2, "X": 2}
    excluded_classes = ("Y", "C")


@viz_experiment.config
def viz_semi_permanent():

    class_code_to_name = {0: "A", 1: "B", 2: "BACKGROUND"}
    name_to_rgb_code = {"A": "#B65B47", "B": "#A36DE9", "BACKGROUND": "#AAAAAA"}
    visualization_columns = 600


@shuffle_experiment.config
def shuffle_semi_permanent():

    train_pct = 0.8
    extension = ".pkl"

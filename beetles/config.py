import yaml
import os


class Config:

    default_model_directory = os.path.join(os.path.expanduser("~"), ".cache", "beetles")

    name_to_index = {"A": 0, "B": 1, "BACKGROUND": 2}

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

    mask_flag = -1
    excluded_classes = ("Y", "C")

    class_code_to_name = {0: "A", 1: "B", 2: "BACKGROUND"}

    sound_type_to_color = {"A": "r", "B": "y", "BACKGROUND": "k"}
    name_to_rgb_code = {"A": "#b65b47", "B": "#A36DE9", "BACKGROUND": "#AAAAAA"}
    aws_download_link = "https://beetles-cnn-models.s3.amazonaws.com/model_{}.pt"
    default_spectrogram_num_rows = 128
    visualization_n_fft = 1150
    vertical_cut = 20

    key_to_label = {"y": "A", "w": "B", "e": "BACKGROUND"}
    label_keys = set(key_to_label.keys())

    def __init__(self, config_file=None):

        self.config_file = config_file

        if self.config_file is not None:

            with open(self.config_file, "r") as src:
                dct = yaml.safe_load(src)
            for k, v in dct.items():
                setattr(self, k, v)

    def __str__(self):
        ls = []
        for k, v in self.__dict__.items():
            ls.append(f"{k}: {v}")
        return "  ".join(ls)

    @property
    def index_to_name(self):
        return {v: k for k, v in self.name_to_index.items()}

    @property
    def name_to_class_code(self):
        return {v: k for k, v in self.class_code_to_name.items()}

    def __getitem__(self, item):
        return getattr(self, item)

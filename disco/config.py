import yaml
import os


class Config:
    """
    Class that contains all configuration variables for this project.
    If a .yaml file is passed in upon creation, it'll parse the yaml file
    and overwrite default config values with those in the yaml file.
    See ./resources/example_beetles_config.yaml or the wiki for more information about each of these variables.
    """

    name_to_class_code = {"A": 0, "B": 1, "BACKGROUND": 2, "X": 2}
    class_code_to_name = {0: "A", 1: "B", 2: "BACKGROUND"}

    name_to_rgb_code = {"A": "#B65B47", "B": "#A36DE9", "BACKGROUND": "#AAAAAA"}
    visualization_n_fft = 1150
    visualization_columns = 1000

    excluded_classes = ("Y", "C")

    default_model_directory = os.path.join(os.path.expanduser("~"), ".cache", "disco")

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

    aws_download_link = "https://disco-models.s3.us-west-1.amazonaws.com/random_init_model_{}.ckpt"

    vertical_cut = 20
    key_to_label = {"y": "A", "w": "B", "e": "BACKGROUND"}

    mask_flag = -1
    default_spectrogram_num_rows = 128

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

    def __getitem__(self, item):
        return getattr(self, item)

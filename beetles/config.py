import yaml
import os

class Config:
    
    def __init__(self, config_file=None):
        
        self.default_model_directory = os.path.join(os.path.expanduser("~"), ".cache", "beetles")
        
        self.name_to_index = {"A": 0,
                              "B": 1,
                              "BACKGROUND": 2} 
        self.index_to_name = {0: "A",
                              1: "B",
                              2: "BACKGROUND"} 
        
        self.hmm_transition_probabilities = [
            [0.995, 0.00000, 0.005],
            [0.0000, 0.995, 0.005],
            [0.00001, 0.00049, 0.9995]
        ]
        self.hmm_start_probabilities = [0, 0, 1]
        self.hmm_emission_probabilities = [
            {0: 0.995, 1: 0.00005, 2: 0.00495},
            {0: 0.1,   1: 0.88,    2: 0.020},
            {0: 0.05,  1:  0.05,   2: 0.9}
        ]
        
        self.mask_flag = -1
        self.excluded_classes = ("Y", "C")
        
        self.class_code_to_name = {0: "A", 1: "B", 2: "BACKGROUND"}
        self.name_to_class_code = {v: k for k, v in self.class_code_to_name.items()}
        self.sound_type_to_color = {"A": "r", "B": "y", "BACKGROUND": "k"}
        self.aws_download_link = "https://beetles-cnn-models.s3.amazonaws.com/model_{}.pt"
        self.default_spectrogram_num_rows = 128
        
        self.key_to_label = {"y": "A", "w": "B", "e": "BACKGROUND"}
        self.label_keys = set(self.key_to_label.keys())
        
    def __getitem__(self, item):
        return self.key_to_label[item]

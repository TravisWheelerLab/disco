import os

from disco_sound.cfg import extract_experiment


@extract_experiment.config
def config():

    wav_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav_file = os.path.join(wav_file, "resources", "example.wav")
    csv_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(csv_file, "resources", "example_labels.csv")
    output_data_path = "/tmp/extracted_test/"

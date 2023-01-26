import os

from disco_sound.cfg import label_experiment


@label_experiment.config
def config():
    wav_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav_file = os.path.join(wav_file, "resources", "example.wav")
    output_csv_path = "/tmp/test.csv"

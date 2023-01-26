from disco_sound.cfg import label_experiment


@label_experiment.config
def config():
    wav_file = "disco_sound/resources/example.wav"
    output_csv_path = "/tmp/test.csv"

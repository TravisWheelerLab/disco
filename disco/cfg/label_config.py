from disco.cfg import label_experiment


@label_experiment.config
def config():
    wav_file = "disco/resources/example.wav"
    output_csv_path = "/tmp/test.csv"

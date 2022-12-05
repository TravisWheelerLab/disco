from sacred import Experiment

label_experiment = Experiment()


@label_experiment.config
def config():
    wav_file = "/Users/mac/share/disco/disco/resources/example.wav"
    output_csv_path = "/tmp/fork"

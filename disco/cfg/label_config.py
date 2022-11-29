from sacred import Experiment

label_experiment = Experiment()


@label_experiment.config
def config():
    wav_file = None
    output_csv_path = None

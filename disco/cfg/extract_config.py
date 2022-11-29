from sacred import Experiment

extract_experiment = Experiment()


@extract_experiment.config
def config():

    csv_file = None
    wav_file = None
    output_data_path = None

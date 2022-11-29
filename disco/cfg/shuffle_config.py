from sacred import Experiment

shuffle_experiment = Experiment()


@shuffle_experiment.config
def config():
    data_directory = None
    move = False

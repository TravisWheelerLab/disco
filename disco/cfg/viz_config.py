from sacred import Experiment

viz_experiment = Experiment()


@viz_experiment.config
def config():

    data_path = "/Users/mac/share/disco/disco/resources/example-viz"
    medians = True
    post_process = False
    means = False
    iqr = False
    votes = False
    votes_line = False
    second_data_path = None

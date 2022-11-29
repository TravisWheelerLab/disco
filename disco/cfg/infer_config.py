from sacred import Experiment

infer_experiment = Experiment()


@infer_experiment.config
def config():

    wav_file = "/Users/wheelerlab/share/disco/disco/resources/example.wav"
    output_csv_path = "/tmp/fuck_you.csv"
    metrics_path = None
    viz_path = "/tmp/testing"
    accuracy_metrics = False
    accuracy_metrics_test_directory = None

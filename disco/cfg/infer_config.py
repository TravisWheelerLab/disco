from sacred import Experiment

from disco.util.util import to_dict

infer_experiment = Experiment()


@infer_experiment.config
def config():
    wav_file = "/Users/mac/share/disco/disco/resources/example.wav"
    model_name = "UNet1D"
    dataset_name = "SpectrogramIterator"
    saved_model_directory = None
    output_directory = None

    @to_dict
    class dataloader_args:
        snr = 0

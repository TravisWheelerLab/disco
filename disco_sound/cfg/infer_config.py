import os

from disco_sound.cfg import infer_experiment
from disco_sound.util.util import to_dict


@infer_experiment.config
def config():

    wav_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav_file = os.path.join(wav_file, "resources", "example.wav")
    model_name = "UNet1D"
    dataset_name = "SpectrogramIterator"
    saved_model_directory = None
    output_directory = None

    @to_dict
    class dataloader_args:
        vertical_trim = 20
        tile_size = 1024
        tile_overlap = 128
        n_fft = 1150
        hop_length = 200
        log_spect = (True,)
        mel_transform = (True,)

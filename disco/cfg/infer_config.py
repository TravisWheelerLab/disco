from sacred import Experiment

from disco.util.util import to_dict

infer_experiment = Experiment()


@infer_experiment.config
def config():
    wav_file = "/xdisk/twheeler/colligan/ground_truth/180101_0183S34D06.wav"
    model_name = "UNet1D"
    dataset_name = "SpectrogramIterator"
    saved_model_directory = None
    output_directory = None

    @to_dict
    class dataloader_args:
        snr = 0
        vertical_trim = 20
        tile_size = 1024
        tile_overlap = 128
        n_fft = 1150
        hop_length = 200
        log_spect = (True,)
        mel_transform = (True,)
        add_beeps = False

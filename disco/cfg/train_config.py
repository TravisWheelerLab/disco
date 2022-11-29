from glob import glob

from sacred import Experiment

from disco.util.util import to_dict

train_experiment = Experiment()


@train_experiment.config
def config():
    model_name = "UNet1D"
    log_dir = "/tmp/testing"
    dataset_name = "SpectrogramDatasetMultiLabel"

    @to_dict
    class model_args:
        in_channels = 108
        out_channels = 3
        learning_rate = 0.0001
        mel = True
        apply_log = True
        n_fft = 1150
        vertical_trim = 20
        mask_beginning_and_end = (True,)
        begin_mask = 28
        end_mask = 10
        mask_character = -1

    @to_dict
    class train_dataset_args:
        files = glob("/some/path")
        apply_log = True
        vertical_trim = 20
        mask_beginning_and_end = (True,)
        bootstrap_sample = False
        begin_mask = 28
        end_mask = 10

    @to_dict
    class val_dataset_args:
        files = glob("/some/path")
        apply_log = True
        vertical_trim = 20
        mask_beginning_and_end = (True,)
        bootstrap_sample = False
        begin_mask = 28
        end_mask = 10

    @to_dict
    class dataloader_args:
        batch_size = 32
        num_workers = 1

    @to_dict
    class trainer_args:
        gpus = 0

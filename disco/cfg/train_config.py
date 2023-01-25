from glob import glob

from sacred import Experiment

from disco.util.util import to_dict

train_experiment = Experiment()


def beetles_config():
    model_name = "UNet1D"
    log_dir = "/xdisk/twheeler/colligan/more_ensembles/bootstrap"
    dataset_name = "SpectrogramDatasetMultiLabel"

    @to_dict
    class model_args:
        in_channels = 108
        out_channels = 3
        learning_rate = 0.001
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
        files = glob(
            "/xdisk/twheeler/colligan/disco/extracted_data_1150_nffts/train/*pkl"
        )
        vertical_trim = 20
        bootstrap_sample = True

    @to_dict
    class val_dataset_args:
        files = glob(
            "/xdisk/twheeler/colligan/disco/extracted_data_1150_nffts/validation/*pkl"
        )
        vertical_trim = 20

    @to_dict
    class dataloader_args:
        batch_size = 32
        num_workers = 8

    @to_dict
    class trainer_args:
        accelerator = "gpu"
        devices = 1
        max_epochs = 100


@train_experiment.config
def whale_config():
    model_name = "WhaleUNet"
    log_dir = "/xdisk/twheeler/colligan/whale_models_jan17/"
    dataset_name = "WhaleDataset"

    @to_dict
    class model_args:
        in_channels = 128
        out_channels = 1
        learning_rate = 0.001
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
        files = glob("/xdisk/twheeler/colligan/whale_data/data/train/*")
        files = files[: int(0.8 * len(files))]

        label_csv = "/xdisk/twheeler/colligan/whale_data/data/train.csv"
        n_fft = 1150
        hop_length = 20

    @to_dict
    class val_dataset_args:
        files = glob("/xdisk/twheeler/colligan/whale_data/data/train/*")
        files = files[int(0.8 * len(files)) :]
        label_csv = "/xdisk/twheeler/colligan/whale_data/data/train.csv"
        n_fft = 1150
        hop_length = 20

    @to_dict
    class dataloader_args:
        batch_size = 256
        num_workers = 8

    @to_dict
    class trainer_args:
        gpus = 1
        max_epochs = 100

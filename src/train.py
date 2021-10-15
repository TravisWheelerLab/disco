from pytorch_lightning import seed_everything
from time import time

seed_everything(int(time.time()*1000))
import os
import torch
import pytorch_lightning as pl
import spectrogram_analysis as sa
from pytorch_lightning.plugins import DDPPlugin

from models import CNN1D
from argparse import ArgumentParser
from dataset import SpectrogramDataset

DEFAULT_SPECTROGRAM_NUM_ROWS = 128


def train_parser():
    ap = ArgumentParser('train routine')

    tunable = ap.add_argument_group(description='tunable args')
    tunable.add_argument('--n_fft', type=int, required=True)
    tunable.add_argument('--learning_rate', type=float, required=True)
    tunable.add_argument('--begin_cutoff_idx', type=int, required=True)
    tunable.add_argument('--vertical_trim', type=int, required=True)

    non_tunable = ap.add_argument_group(description='non-tunable args')
    non_tunable.add_argument('--log', action='store_true')
    non_tunable.add_argument('--mel', action='store_true')
    non_tunable.add_argument('--bootstrap', action='store_true')
    non_tunable.add_argument('--batch_size', type=int, required=False)
    non_tunable.add_argument('--tune_initial_lr', action='store_true')
    non_tunable.add_argument('--gpus', type=int, required=False)
    non_tunable.add_argument('--num_nodes', type=int, required=False)
    non_tunable.add_argument('--epochs', type=int, required=False)
    non_tunable.add_argument('--check_val_every_n_epoch', type=int, required=False,
                             default=1)
    non_tunable.add_argument('--log_dir', type=str, required=False)
    non_tunable.add_argument('--data_path', type=str, required=False)
    non_tunable.add_argument('--model_name', type=str, default='model.pt')
    non_tunable.add_argument('--num_workers', type=int, required=False)

    return ap


def train_func(hparams):
    spect_type = 'mel_no_log_{}_no_vert_trim'.format(hparams.n_fft)

    train_dataset = SpectrogramDataset('train',
                                       spect_type=spect_type,
                                       data_path=hparams.data_path,
                                       apply_log=hparams.log,
                                       vertical_trim=hparams.vertical_trim,
                                       begin_cutoff_idx=hparams.begin_cutoff_idx,
                                       bootstrap_sample=hparams.bootstrap,
                                       clip_spects=True)

    test_dataset = SpectrogramDataset(dataset_type='test',
                                      data_path=hparams.data_path,
                                      spect_type=spect_type,
                                      apply_log=hparams.log,
                                      vertical_trim=hparams.vertical_trim,
                                      begin_cutoff_idx=hparams.begin_cutoff_idx,
                                      clip_spects=False,
                                      bootstrap_sample=False)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=hparams.batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=hparams.num_workers)
    # batch size of 1 since we're evaluating on lots of differently sized
    # labeled regions. Could implement a mask to zero out bits of predictions
    # but this is too much work for the amount of time it takes to evaluate our validation
    # set.
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=hparams.num_workers)

    # i want to finetune on large spectrograms with multiple different classes so
    # context is learned

    in_channels = DEFAULT_SPECTROGRAM_NUM_ROWS - hparams.vertical_trim

    model = CNN1D(in_channels,
                  hparams.learning_rate,
                  hparams.mel,
                  hparams.log,
                  hparams.n_fft,
                  hparams.begin_cutoff_idx,
                  hparams.vertical_trim)

    save_best = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{val_loss:.5f}',
        mode='min',
        save_top_k=5)

    trainer_kwargs = {
        'gpus': hparams.gpus,
        'num_nodes': hparams.num_nodes,
        'max_epochs': hparams.epochs,
        'check_val_every_n_epoch': hparams.check_val_every_n_epoch,
        'callbacks': [save_best],
        'accelerator': 'ddp' if hparams.gpus else None,
        'plugins': DDPPlugin(find_unused_parameters=False) if hparams.gpus else None,
        'precision': 16 if hparams.gpus else 32,
        'default_root_dir': hparams.log_dir,
        'log_every_n_steps': 1
    }

    if hparams.tune_initial_lr:
        trainer_kwargs['auto_lr_find'] = True

        trainer = pl.Trainer(
            **trainer_kwargs
        )

        trainer.tune(model, train_loader, test_loader)

    else:
        trainer = pl.Trainer(
            **trainer_kwargs
        )

    trainer.fit(model, train_loader, test_loader)

    torch.save(model.state_dict(), os.path.join(trainer.log_dir, hparams.model_name))


if __name__ == '__main__':
    args = train_parser().parse_args()
    train_func(args)

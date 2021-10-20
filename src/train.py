from pytorch_lightning import seed_everything
from time import time

# BE CAREFUL. This will cause inconsistent behavior if training
# using DDP!
seed_everything(int(time() * 1000))

import os
import torch
import pytorch_lightning as pl
import spectrogram_analysis as sa
from pytorch_lightning.plugins import DDPPlugin
from glob import glob

from models import SimpleCNN, UNet1D
from argparse import ArgumentParser
from dataset import SpectrogramDatasetMultiLabel, pad_batch

DEFAULT_SPECTROGRAM_NUM_ROWS = 128


def train_parser():
    ap = ArgumentParser('train routine')

    tunable = ap.add_argument_group(title='tunable args', description='arguments in this group are'
                                                                      'tunable, and tuned in hparam_optimizer.py')
    tunable.add_argument('--n_fft', type=int, required=True,
                         help='number of ffts used to create the spectrogram')
    tunable.add_argument('--learning_rate', type=float, required=True,
                         help='initial learning rate')
    tunable.add_argument('--vertical_trim', type=int, required=True,
                         help='how many rows to remove from the low-frequency range of the spectrogram.'
                              'This is probably unnecessary because NaNs are easily removed in preprocessing.')
    tunable.add_argument('--mask_beginning_and_end', action='store_true',
                         help='whether or not to mask the beginning and end of single-label chirps')
    tunable.add_argument('--begin_mask', type=int, default=30,
                         help='how many cols to mask from beginning')
    tunable.add_argument('--end_mask', type=int, default=10,
                         help='how many cols to mask from end')
    non_tunable = ap.add_argument_group(title='non-tunable args', description='the "mel" argument depends on the data'
                                                                              'extraction step - whether or not a mel'
                                                                              'spectrogram was computed')
    non_tunable.add_argument('--log', action='store_true', help='whether or not to apply a log2 transform to the'
                                                                'spectrogram')
    non_tunable.add_argument('--mel', action='store_true', help='use a mel-transformed spectrogram')
    non_tunable.add_argument('--bootstrap', action='store_true', help='train a model with a sample of the training set'
                                                                      '(replace=True)')
    non_tunable.add_argument('--batch_size', type=int, required=True, help='batch size')
    non_tunable.add_argument('--tune_initial_lr', action='store_true', help='whether or not to use PyTorchLightning\'s'
                                                                            'built-in initial LR tuner')
    non_tunable.add_argument('--gpus', type=int, required=True, help='number of gpus per node')
    non_tunable.add_argument('--num_nodes', type=int, required=True, help='number of nodes. If you want to train with 8'
                                                                          'GPUs, --gpus should be 4 and --num_nodes'
                                                                          'should be 2 (assuming you have 4 GPUs per '
                                                                          'node')
    non_tunable.add_argument('--epochs', type=int, required=True, help='max number of epochs to train')
    non_tunable.add_argument('--check_val_every_n_epoch', type=int, required=False,
                             default=1, help='how often to validate the model. On each validation run the loss is '
                                             'logged '
                                             'and if it\'s lower than the previous best the current model is saved')
    non_tunable.add_argument('--log_dir', type=str, required=True, help='where to save the model logs (train, test '
                                                                        'loss '
                                                                        'and hyperparameters). Visualize with '
                                                                        'tensorboard')
    non_tunable.add_argument('--data_path', type=str, required=True, help='where the data are saved on disk. Assumes'
                                                                          'the data were saved with np.save and reside'
                                                                          'in <test/train/validation>/spect/*npy')
    non_tunable.add_argument('--model_name', type=str, default='model.pt', help='custom model name for saving the model'
                                                                                'after training has completed')
    non_tunable.add_argument('--num_workers', type=int, required=True,
                             help='number of threads to use when loading data')

    return ap


def train_func(hparams):
    # TODO: Don't hardcode this.

    spect_type = 'mel_no_log_{}_no_vert_trim'.format(hparams.n_fft)
    print(spect_type)

    train_files = glob(os.path.join(hparams.data_path, 'train', spect_type, "*"))
    val_files = glob(os.path.join(hparams.data_path, 'validation', spect_type, "*"))

    train_dataset = SpectrogramDatasetMultiLabel(train_files,
                                                 apply_log=hparams.log,
                                                 vertical_trim=hparams.vertical_trim,
                                                 bootstrap_sample=hparams.bootstrap,
                                                 mask_beginning_and_end=hparams.mask_beginning_and_end,
                                                 begin_mask=hparams.begin_mask,
                                                 end_mask=hparams.end_mask)

    val_dataset = SpectrogramDatasetMultiLabel(val_files,
                                               apply_log=hparams.log,
                                               vertical_trim=hparams.vertical_trim,
                                               bootstrap_sample=False,
                                               mask_beginning_and_end=False,
                                               begin_mask=None,
                                               end_mask=None)
    # batch size of 1 since we're evaluating on lots of differently sized
    # labeled regions. Could implement a mask to zero out bits of predictions
    # but this is too much work for the amount of time it takes to evaluate our validation
    # set.
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=hparams.batch_size,
                                               shuffle=True,
                                               num_workers=hparams.num_workers,
                                               collate_fn=None if hparams.batch_size == 1 else pad_batch)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=hparams.batch_size,
                                             shuffle=False,
                                             num_workers=hparams.num_workers,
                                             collate_fn=None if hparams.batch_size == 1 else pad_batch)

    in_channels = DEFAULT_SPECTROGRAM_NUM_ROWS - hparams.vertical_trim

    # TODO: refactor this to incorporate data loaders more easily.
    model = UNet1D(in_channels,
                   hparams.learning_rate,
                   hparams.mel,
                   hparams.log,
                   hparams.n_fft,
                   hparams.vertical_trim,
                   hparams.mask_beginning_and_end,
                   hparams.begin_mask,
                   hparams.end_mask,
                   train_dataset.files,
                   val_dataset.files)

    save_best = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{val_loss:.5f}',
        mode='min',
        save_top_k=1)

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
        'log_every_n_steps': 1,
        'terminate_on_nan': True,
    }

    if hparams.tune_initial_lr:
        trainer_kwargs['auto_lr_find'] = True

        trainer = pl.Trainer(
            **trainer_kwargs
        )

        trainer.tune(model, train_loader, val_loader)

    else:
        trainer = pl.Trainer(
            **trainer_kwargs
        )

    trainer.fit(model, train_loader, val_loader)

    torch.save(model.state_dict(), os.path.join(trainer.log_dir, hparams.model_name))


if __name__ == '__main__':
    args = train_parser().parse_args()
    train_func(args)

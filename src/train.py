import os
import torch
import pytorch_lightning as pl
import spectrogram_analysis as sa
from pytorch_lightning.plugins import DDPPlugin

from models import CNN1D
from argparse import ArgumentParser
from dataset import SpectrogramDataset


def parser():
    ap = ArgumentParser()
    ap.add_argument('--vert_trim', required=True, type=int)
    ap.add_argument('--n_fft', required=True, type=int)
    ap.add_argument('--log', action='store_true')
    ap.add_argument('--mel', action='store_true')
    ap.add_argument('--bootstrap', action='store_true')
    ap.add_argument('--batch_size', type=int, required=True)
    ap.add_argument('--tune_initial_lr', action='store_true')
    ap.add_argument('--in_channels', type=int, required=True)
    ap.add_argument('--learning_rate', type=float, required=True)
    ap.add_argument('--gpus', type=int, required=True)
    ap.add_argument('--num_nodes', type=int, required=True)
    ap.add_argument('--epochs', type=int, required=True)
    ap.add_argument('--check_val_every_n_epoch', type=int, required=False,
                    default=1)
    ap.add_argument('--log_dir', type=str, required=True)
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model_name', type=str, default='model.pt')
    ap.add_argument('--num_workers', type=int, required=True)

    return ap.parse_args()


def main(hparams):
    if hparams.vert_trim is None:
        vert_trim = sa.determine_default_vert_trim(hparams.mel, hparams.log, hparams.n_fft)

    spect_type = sa.form_spectrogram_type(hparams.mel, hparams.n_fft, hparams.log, hparams.vert_trim)

    train_dataset = SpectrogramDataset('train',
                                       spect_type=spect_type,
                                       data_path=hparams.data_path,
                                       bootstrap_sample=hparams.bootstrap,
                                       clip_spects=True)

    test_dataset = SpectrogramDataset(dataset_type='test',
                                      data_path=hparams.data_path,
                                      spect_type=spect_type,
                                      clip_spects=False,
                                      bootstrap_sample=False)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=hparams.batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=hparams.num_workers)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=hparams.num_workers)
    # i want to finetune on large regions

    model = CNN1D(hparams.in_channels,
                  hparams.learning_rate,
                  hparams.mel,
                  hparams.log,
                  hparams.n_fft)

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
        'accelerator': 'ddp',
        'plugins': DDPPlugin(find_unused_parameters=False),
        'precision': 16,
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
    main(parser())

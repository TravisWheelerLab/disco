from argparse import ArgumentParser

import torch

from inference_utils import SpectrogramIterator


def parser():
    ap = ArgumentParser()
    ap.add_argument('--saved_model_directory', required=True, type=str,
                    help='where the ensemble of models is stored')
    ap.add_argument('--model_extension', default='.pt', type=str,
                    help='filename extension of saved model files')
    ap.add_argument('--wav_file', required=True, type=str,
                    help='.wav file to predict')
    ap.add_argument('--output_csv_path', required=True, type=str,
                    help='where to save the final predictions')
    ap.add_argument('--tile_overlap', default=32, type=int,
                    help='how much to overlap consecutive predictions')
    ap.add_argument('--tile_size', default=256, type=int,
                    help='model input size')
    ap.add_argument('--batch_size', default=32, type=int,
                    help='batch size')
    return ap.parse_args()


def main(args):
    # how will this go?
    # overlap-tile the spectrogram in dataloader
    tile_overlap = 32
    x = SpectrogramIterator(256,
                            tile_overlap,
                            args.wav_file,
                            vertical_trim=20,
                            n_fft=900,
                            hop_length=200,
                            log_spect=True,
                            mel_transform=True)

    dataset = torch.utils.data.DataLoader(x,
                                          shuffle=False,
                                          batch_size=args.batch_size,
                                          drop_last=False)
    # now I need to splice together the tiled chunks of
    # spectrogram (do this after each model has predicted
    # the batch).
    # i = 0
    # all = []
    # for d in dataset:
    #     stitched_together = [z[:, tile_overlap:-tile_overlap] for z in d]
    #     a = torch.cat(stitched_together, dim=-1)
    #     all.append(a.squeeze())
    # stitched_together = torch.cat(all, dim=1)[:, :x.original_shape[-1]]
    # print(stitched_together.shape, x.original_shape)
    # done, this works now.
    # now evaluate with models.


if __name__ == '__main__':
    main(parser())

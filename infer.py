import pdb
import warnings
import numpy as np
import os
import torch

from argparse import ArgumentParser

import heuristics as heuristics
import inference_utils as infer

# get rid of torchaudio warning us that our spectrogram calculation needs different parameters
warnings.filterwarnings("ignore", category=UserWarning)


def parser():
    ap = ArgumentParser()
    ap.add_argument('--saved_model_directory', required=True, type=str,
                    help='where the ensemble of models is stored')
    ap.add_argument('--model_extension', default='.pt', type=str,
                    help='filename extension of saved model files')
    ap.add_argument('--wav_file', required=True, type=str,
                    help='.wav file to predict')
    ap.add_argument('--output_csv_path', default=None, type=str,
                    help='where to save the final predictions')
    ap.add_argument('--tile_overlap', default=32, type=int,
                    help='how much to overlap consecutive predictions. Larger values will mean slower performance as '
                         'there is more repeated computation')
    ap.add_argument('--tile_size', default=256, type=int,
                    help='length of input spectrogram')
    ap.add_argument('--batch_size', default=32, type=int,
                    help='batch size')
    ap.add_argument('--input_channels', default=98, type=int,
                    help='number of channels of input spectrogram')
    ap.add_argument('--plot_prefix', type=str, default=None,
                    help='where to save the median prediction plot')
    ap.add_argument('--hop_length', type=int, default=200,
                    help='length of hops b/t subsequent spectrogram windows')

    args = ap.parse_args()
    return args


def main(args):
    if args.tile_size % 2 != 0:
        # todo: fix this bug
        raise ValueError('tile_size must be even, got {}'.format(args.tile_size))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = infer.assemble_ensemble(args.saved_model_directory, args.model_extension, device, args.input_channels)

    # doesn't work with an uneven tile_size for some reason - it doesn't pad the spectrogram enough
    spectrogram_iterator = infer.SpectrogramIterator(args.tile_size,
                                                     args.tile_overlap,
                                                     args.wav_file,
                                                     vertical_trim=30,
                                                     n_fft=800,
                                                     hop_length=200,
                                                     log_spect=True,
                                                     mel_transform=True)

    spectrogram_dataset = torch.utils.data.DataLoader(spectrogram_iterator,
                                                      shuffle=False,
                                                      batch_size=args.batch_size,
                                                      drop_last=False)

    # Need to predict our final test dataset with the ensemble.
    # todo
    medians, iqr = infer.evaluate_spectrogram(spectrogram_dataset,
                                              models,
                                              spectrogram_iterator.tile_overlap,
                                              spectrogram_iterator.original_shape,
                                              device=device)

    # TODO: add pre-and-post processing heuristics based on IQR.
    predictions = np.argmax(medians, axis=0).squeeze()
    predictions = heuristics.threshold_as_on_iqr(predictions, iqr, 0.5)

    hmm_predictions = infer.run_hmm(predictions)

    if args.output_csv_path is not None:
        # I need to figure out how to convert spectrogram index to .wav index.
        # it'll require a multiplication since each spectrogram index is actually
        # 200 or so
        # default window size is n_fft.
        infer.save_csv_from_predictions(args.output_csv_path,
                                        hmm_predictions,
                                        sample_rate=spectrogram_iterator.sample_rate,
                                        hop_length=args.hop_length)

    if args.plot_prefix is not None:
        plot_prefix = args.plot_prefix + os.path.splitext(os.path.basename(args.wav_file))[0]
        infer.plot_predictions_and_confidences(spectrogram_iterator.original_spectrogram,
                                               medians,
                                               iqr,
                                               hmm_predictions,
                                               predictions,
                                               plot_prefix,
                                               n_samples=30)


if __name__ == '__main__':
    main(parser())

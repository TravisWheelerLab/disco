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
    ap.add_argument('--vertical_trim', type=int, default=30,
                    help='how many rows to remove from the spectrogram ')
    ap.add_argument('--n_fft', type=int, default=800,
                    help='size of the fft to use when calculating spectrogram')
    ap.add_argument('--debug', action='store_true',
                    help='whether or not to save debug information for inspection with interactive_plot.py')
    ap.add_argument('--n_images', type=int, default=30,
                    help='how many images to save when plot_prefix is specified')
    ap.add_argument('--len_image_sample', type=int, default=1000,
                    help='how long each image should be')
    args = ap.parse_args()

    return args


def main(args):
    if args.tile_size % 2 != 0:
        raise ValueError('tile_size must be even, got {}'.format(args.tile_size))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = infer.assemble_ensemble(args.saved_model_directory, args.model_extension, device, args.input_channels)
    if len(models) < 2:
        raise ValueError('expected more than 1 model, found {}. Is the model directory and extension correct?'.format(len(models)))

    spectrogram_iterator = infer.SpectrogramIterator(args.tile_size,
                                                     args.tile_overlap,
                                                     args.wav_file,
                                                     vertical_trim=args.vertical_trim,
                                                     n_fft=args.n_fft,
                                                     hop_length=args.hop_length,
                                                     log_spect=True,
                                                     mel_transform=True)

    spectrogram_dataset = torch.utils.data.DataLoader(spectrogram_iterator,
                                                      shuffle=False,
                                                      batch_size=args.batch_size,
                                                      drop_last=False)

    medians, iqr = infer.evaluate_spectrogram(spectrogram_dataset,
                                              models,
                                              spectrogram_iterator.tile_overlap,
                                              spectrogram_iterator.original_shape,
                                              device=device)

    predictions = np.argmax(medians, axis=0).squeeze()
    for heuristic in heuristics.HEURISTIC_FNS:
        predictions = heuristic(predictions, iqr)

    hmm_predictions = infer.run_hmm(predictions)

    if args.output_csv_path is not None:
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
                                               n_images=args.n_images,
                                               len_sample=args.len_image_sample)
    if args.debug is not None:

        debug_path = './debug'
        if not os.path.isdir(debug_path):
            os.mkdir(debug_path)

        spectrogram_path = os.path.join(debug_path,       'raw_spectrogram.pkl')
        hmm_prediction_path = os.path.join(debug_path,    'hmm_predictions.pkl')
        median_prediction_path = os.path.join(debug_path, 'median_predictions.pkl')

        iqr_path = os.path.join(debug_path,               'iqrs.pkl')
        csv_path = os.path.join(debug_path,               'classifications.csv')

        infer.save_csv_from_predictions(csv_path,
                                        hmm_predictions,
                                        sample_rate=spectrogram_iterator.sample_rate,
                                        hop_length=args.hop_length)

        infer.pickle_data(spectrogram_iterator.original_spectrogram, spectrogram_path)
        infer.pickle_data(hmm_predictions, hmm_prediction_path)
        infer.pickle_data(medians, median_prediction_path)
        infer.pickle_data(iqr, iqr_path)


if __name__ == '__main__':
    main(parser())

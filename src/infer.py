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

# if there isn't an Xwindow server

def parser():

    ap = ArgumentParser()
    ap.add_argument('--saved_model_directory', required=False,
                    default=infer.DEFAULT_MODEL_DIRECTORY,
                    type=str, help='where the ensemble of models is stored')
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
    ap.add_argument('--hop_length', type=int, default=200,
                    help='length of hops b/t subsequent spectrogram windows')
    ap.add_argument('--vertical_trim', type=int, default=30,
                    help='how many rows to remove from the spectrogram ')
    ap.add_argument('--n_fft', type=int, default=800,
                    help='size of the fft to use when calculating spectrogram')
    ap.add_argument('--debug', type=str, default=None,
                    help='where to save debugging data')
    ap.add_argument('--num_threads', type=int, default=4,
                    help='how many threads to use when evaluating on CPU')
    args = ap.parse_args()

    return args


def main(args):

    if args.tile_size % 2 != 0:
        raise ValueError('tile_size must be even, got {}'.format(args.tile_size))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        torch.set_num_threads(args.num_threads)
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
                                              args.tile_overlap,
                                              spectrogram_iterator.original_shape,
                                              device=device)

    predictions = np.argmax(medians, axis=0).squeeze()
    for heuristic in heuristics.HEURISTIC_FNS:
        print('applying heuristic function', heuristic.__name__)
        predictions = heuristic(predictions, iqr)

    hmm_predictions = infer.smooth_predictions_with_hmm(predictions)

    if args.output_csv_path is not None:
        infer.save_csv_from_predictions(args.output_csv_path,
                                        hmm_predictions,
                                        sample_rate=spectrogram_iterator.sample_rate,
                                        hop_length=args.hop_length)

    if args.debug is not None:

        debug_path = args.debug
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

import pdb
import warnings
import numpy as np
import os
import torch

from argparse import ArgumentParser

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

    # TODO: add pre-and-post processing based on IQR.
    predictions = np.argmax(medians, axis=0).squeeze()
    summed_iqr = np.sum(iqr, axis=0)
    normalized_summed_iqr = summed_iqr / np.max(summed_iqr[predictions == infer.NAME_TO_CLASS_CODE['A']])
    # We've noticed that true As and Bs are often predicted correctly
    # with the ensemble. Background is sometimes classified as A by the ensemble but with
    # a high uncertainty. Below we force A predictions with high uncertainty to background.
    predictions[
        (normalized_summed_iqr >= 0.05).astype(bool) & (predictions == infer.NAME_TO_CLASS_CODE['A']).astype(bool)] = \
    infer.NAME_TO_CLASS_CODE['BACKGROUND']
    # todo: get confusion matrix on the test data for the model ensemble.

    hmm_predictions = infer.run_hmm(predictions)

    if args.output_csv_path is not None:
        # I need to figure out how to convert spectrogram index to .wav index.
        # it'll require a multiplication since each spectrogram index is actually
        # 200 or so
        # default window size is n_fft.

        class_idx_to_prediction_start_and_end = infer.aggregate_predictions(hmm_predictions)
        import matplotlib.pyplot as plt
        for j in range(3):
            i = 0
            for begin, end in class_idx_to_prediction_start_and_end[j]:
                fig, ax = plt.subplots(nrows=2, figsize=(13, 10))
                begin = begin - 10 if begin > 10 else begin
                ax[0].imshow(spectrogram_iterator.original_spectrogram[:, begin:end+10])
                hmm_slice = hmm_predictions[begin:end+10]
                hmm_rgb = np.zeros((1, len(hmm_slice), 3))
                for class_idx in infer.CLASS_CODE_TO_NAME.keys():
                    hmm_rgb[:, np.where(hmm_slice == class_idx), class_idx] = 1
                ax[1].imshow(hmm_rgb, aspect='auto')
                ax[1].plot([10, len(hmm_slice)-10], [0, 0], 'ko')
                plt.savefig('/home/tc229954/aggregated_predictions/{}_{}.png'.format(infer.CLASS_CODE_TO_NAME[j], i))
                plt.close()
                i += 1
                if i >= 10:
                    break
    exit()

    if args.plot_prefix is not None:
        plot_prefix = args.plot_prefix + os.path.splitext(os.path.basename(args.wav_file))[0]
        infer.plot_predictions_and_confidences(spectrogram_iterator.original_spectrogram,
                                               medians,
                                               iqr,
                                               hmm_predictions,
                                               predictions,
                                               plot_prefix)


if __name__ == '__main__':
    main(parser())

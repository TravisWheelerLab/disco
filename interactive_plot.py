import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import inference_utils as infer
from glob import glob
from argparse import ArgumentParser


def parser():
    ap = ArgumentParser()
    ap.add_argument('--data_path', type=str,
                    required=True)
    ap.add_argument('--sample_rate', type=int, default=48000,
                    help='sample rate of audio recording')
    ap.add_argument('--hop_length', type=int, default=200,
                    help='length of hops b/t subsequent spectrogram windows')

    return ap.parse_args()


def main(data_root, hop_length, sample_rate):

    predictions = infer.load_pickle(data_root + 'median_predictions.pkl')
    predictions = np.expand_dims(np.transpose(predictions), 0)
    spectrogram = infer.load_pickle(data_root + 'raw_spectrogram.pkl')
    hmm_predictions = infer.convert_argmaxed_array_to_rgb(infer.load_pickle(data_root + 'hmm_predictions.pkl'))

    prediction_df = infer.load_prediction_csv(data_root + 'classifications.csv',
                                              hop_length=hop_length,
                                              sample_rate=sample_rate)

    prediction_df = prediction_df.loc[prediction_df['Sound_Type'] != "BACKGROUND", :]

    fig, ax = plt.subplots(nrows=2, figsize=(13, 10))

    iqr = infer.load_pickle(data_root + 'iqrs.pkl')
    argmaxed_predictions = infer.convert_argmaxed_array_to_rgb(predictions.argmax(axis=-1).squeeze())
    iqr = np.expand_dims(np.transpose(iqr), 0)
    prediction_array = np.concatenate((hmm_predictions, argmaxed_predictions, iqr, predictions), axis=0)

    ax[0].imshow(spectrogram, aspect='auto')

    sound_type_to_color = {'A': 'r', 'B': 'g'}
    for cls, point in zip(prediction_df['Sound_Type'], prediction_df['Begin Spect Index']):
        ax[0].plot([point, point], [0, spectrogram.shape[0]], '{}-'.format(sound_type_to_color[cls]))

    for cls, point in zip(prediction_df['Sound_Type'], prediction_df['End Spect Index']):
        ax[0].plot([point, point], [0, spectrogram.shape[0]], '{}-.'.format(sound_type_to_color[cls]))

    ax[1].imshow(prediction_array, aspect='auto', interpolation='nearest')
    ax[1].axis([1, predictions.shape[-1], -0.25, 2.5])
    ax[1].set_ylim([-0.5, 3.5])

    axcolor = 'lightgoldenrodyellow'
    axpos = plt.axes([0.2, 0.0001, 0.65, 0.03], facecolor=axcolor)
    spos = Slider(axpos, 'Pos', 0.0, predictions.shape[1])

    n = 800
    ax[0].set_xlim([0, n])
    ax[1].set_xlim([0, n])

    def update(val):
        pos = spos.val
        ax[1].axis([pos, pos + n, -0.5, 3.5])
        ax[0].axis([pos, pos + n, 0, spectrogram.shape[0]])
        fig.canvas.draw_idle()

    spos.on_changed(update)
    plt.show()


if __name__ == '__main__':
    args = parser()
    main(args.data_path, args.hop_length, args.sample_rate)

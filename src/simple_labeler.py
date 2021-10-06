import pdb
import matplotlib
import matplotlib.pyplot as plt
import inference_utils as infer
import torchaudio
import numpy as np
import pandas as pd
import os

np.random.seed(19680801)

from argparse import ArgumentParser
from matplotlib.widgets import SpanSelector


def parser():
    ap = ArgumentParser()
    ap.add_argument('--wav_file', required=True, type=str,
                    help='which .wav file to analyze')
    ap.add_argument('--output_csv_path', type=str,
                    default='labels.csv',
                    help='where to save the labels')
    return ap.parse_args()


def add_example(label_list, wav_file, begin_idx, end_idx, sound_type,
                hop_length=None, sample_rate=None):
    begin_time = infer.convert_spectrogram_index_to_seconds(begin_idx,
                                                            hop_length=hop_length,
                                                            sample_rate=sample_rate)
    end_time = infer.convert_spectrogram_index_to_seconds(end_idx,
                                                          hop_length=hop_length,
                                                          sample_rate=sample_rate)
    label_list.append({
        'Begin Time (s)': begin_time,
        'End Time (s)': end_time,
        'Sound_Type': sound_type.upper(),
        'Filename': wav_file
    })


if __name__ == '__main__':

    args = parser()

    waveform, sample_rate = infer.load_wav_file(args.wav_file)

    hop_length = 200

    spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                       n_fft=800,
                                                       hop_length=hop_length,
                                                       )(waveform).log2().numpy().squeeze()
    spectrogram = spectrogram[20:, :]

    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))

    n = 0
    xmin = 0
    xmax = 0
    interval = 400
    label_list = []
    ax1.imshow(spectrogram[:, n:n + interval])

    ax1.set_title('Press left mouse button and drag '
                  'to select a region in the top graph '
                  '0 percent through spectrogram')

    def _redraw_ax2(xmin, xmax):
        global n
        ax2.imshow(spectrogram[:, n + xmin: n + xmax])
        ax2.set_title('selected region')
        fig.canvas.draw()

    def _redraw_ax1(start, interval):
        # could be
        ax1.imshow(spectrogram[:, start:start + interval], aspect='auto')
        ax1.set_title('Press left mouse button and drag '
                      'to select a region in the top graph '
                      '{:d} percent through spectrogram'.format(int(100 * start / spectrogram.shape[-1])))
        fig.canvas.draw()


    def onselect(x_min, x_max):
        global n, xmin, xmax
        xmin = x_min
        xmax = x_max
        if (xmax - xmin) >= 2:
            _redraw_ax2(int(xmin), int(xmax))


    def process_keystroke(key):
        global interval, n, xmin, xmax, hop_length, sample_rate

        if key.key in ('y', 'Y'):
            print('A')
            add_example(label_list, args.wav_file, n+xmin, n+xmax, 'A',
                        hop_length=hop_length,
                        sample_rate=sample_rate)
        elif key.key in ('w', 'W'):
            print('B')
            add_example(label_list, args.wav_file, n+xmin, n+xmax, 'B',
                        hop_length=hop_length,
                        sample_rate=sample_rate)
        elif key.key in ('e', 'E'):
            add_example(label_list, args.wav_file, n+xmin, n+xmax, 'background',
                        hop_length=hop_length,
                        sample_rate=sample_rate)
            print('X')
        elif key.key in ('r', 'R'):
            if len(label_list):
                label_list.pop()
                print('deleting last selection')
            else:
                print('empty label list! hit <A, B, X> after selecting a region'
                      ' to add a labeled region to the list'
                      )
        elif key.key in ('a', 'A'):
            print('widening window')
            interval += 10
            _redraw_ax1(n, interval)
        elif key.key in ('t', 'T'):
            print('tightening window')
            interval -= 10
            _redraw_ax1(n, interval)
        elif key.key in ('f', 'F'):
            n = n + interval // 2
            _redraw_ax1(n, interval)
        elif key.key in ('j', 'J'):
            n = n + 10 * interval
            _redraw_ax1(n, interval)
        elif key.key in ('d', 'D'):
            print('reversing')
            n = n - interval // 2
            _redraw_ax1(n, interval)
        else:
            print("unknown value: hit one of a, b, x")


    fig.canvas.mpl_connect('key_press_event', process_keystroke)

    span = SpanSelector(ax1, onselect, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='tab:blue'))

    plt.show()

    label_df = pd.DataFrame.from_dict(label_list)

    if os.path.isfile(args.output_csv_path):
        label_df.to_csv(args.output_csv_path, index=False, mode='a', header=False)
    else:
        label_df.to_csv(args.output_csv_path, index=False, mode='w')

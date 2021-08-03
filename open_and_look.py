import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import spectrogram_analysis as sa
import argparse


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_sound_type', required=True, type=str)
    ap.add_argument('--number_of_samples', required=True, type=int)
    ap.add_argument('--mel_scale', required=True, type=bool)
    ap.add_argument('--log_scale', required=True, type=bool)
    ap.add_argument('--n_fft', required=True, type=int)
    ap.add_argument('--debugger', required=True, type=bool)
    ap.add_argument('--vert_trim', required=False, default=None)
    ap.add_argument('--optional_path_argument', required=False, default=None)
    return ap.parse_args()


if __name__ == '__main__':

    args = parser()

    target_sound_type = args.target_sound_type
    mel = args.mel_scale
    log = args.log_scale
    n_fft = args.n_fft
    vert_trim = args.vert_trim
    debugger = args.debugger
    number_of_samples = args.number_of_samples
    optional_path_argument = args.optional_path_argument

    if vert_trim is None:
        vert_trim = sa.determine_default_vert_trim(mel, log, n_fft)

    if optional_path_argument:
        spect_type = optional_path_argument
    else:
        spect_type = sa.form_spectrogram_type(mel, n_fft, log, vert_trim)

    root = './data/train/' + spect_type + '/spect'

    files = glob(os.path.join(root, "*"))
    files = [f for f in files if target_sound_type in f]

    i = 0
    plt.style.use("dark_background")

    for f in files:
        if i > number_of_samples:
            break
        arr = np.load(f)
        if not log:
            arr = np.log2(arr)

        if debugger:
            breakpoint()
        else:
            plt.imshow(arr)
            plt.colorbar()
            plt.show()
            i += 1
            plt.savefig('image_offload/' + target_sound_type + str(i) + '.png')
            plt.close()
            print('image', str(i), 'saved to offload directory.')

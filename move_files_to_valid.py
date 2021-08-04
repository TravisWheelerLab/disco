from glob import glob
import os
import spectrogram_analysis as sa
from random import shuffle
import argparse

# this script moves half of the test data files to the validation directory.


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mel_scale', required=True, type=bool)
    ap.add_argument('--log_scale', required=True, type=bool)
    ap.add_argument('--n_fft', required=True, type=int)
    ap.add_argument('--vert_trim', required=False, default=None)
    ap.add_argument('--optional_path_argument', required=False, default=None)
    return ap.parse_args()


args = parser()
# mel = args.mel_scale
# log = args.log_scale
# n_fft = args.n_fft
# savefig = args.savefig
# vert_trim = args.vert_trim
# optional_path_argument = args.optional_path_argument

mel = True
log = True
n_fft = 800
vert_trim = 30
optional_path_argument = None

if vert_trim is None:
    vert_trim = sa.determine_default_vert_trim(mel, log, n_fft)

if optional_path_argument:
    spect_type_to_move_into_valid = optional_path_argument
else:
    spect_type_to_move_into_valid = sa.form_spectrogram_type(mel, n_fft, log, vert_trim)

files = glob(os.path.join('data/test/' + spect_type_to_move_into_valid + '/spect', "*npy"))
shuffle(files)

for uid in ['X', 'A', 'B']:
    class_specific_files = [f for f in files if uid in os.path.basename(f)]
    shuffle(class_specific_files)
    valid_set = class_specific_files[:len(class_specific_files)//2]
    for valid in valid_set:
        out_filename = os.path.join('data/validation/' + spect_type_to_move_into_valid + '/spect/',
                                    os.path.basename(valid))
        os.rename(valid, out_filename)
        print(valid + ' moved.')
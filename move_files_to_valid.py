from glob import glob
import os
import spectrogram_analysis as sa
from random import shuffle

# this script moves half of the test data files to a validation directory.

mel = True
log = True
n_fft = 1600
vert_trim = None

if vert_trim is None:
    vert_trim = sa.determine_default_vert_trim(mel, log, n_fft)

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
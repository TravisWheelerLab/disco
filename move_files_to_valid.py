from glob import glob
import os

from random import shuffle

go = False

files = glob(os.path.join('./test_data/spect', "*npy"))
shuffle(files)

for uid in ['X', 'A', 'B']:
    class_specific_files = [f for f in files if uid in os.path.basename(f)]
    shuffle(class_specific_files)
    valid_set = class_specific_files[:len(class_specific_files)//2]
    for valid in valid_set:
        out_filename = os.path.join('validation_data/spect/', os.path.basename(valid))
        print(valid, out_filename)
        if go:
            os.rename(valid, out_filename)
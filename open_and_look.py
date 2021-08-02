import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import spectrogram_analysis as sa


if __name__ == '__main__':
    target = 'A'
    mel = True
    log = True
    n_fft = 1600
    number_of_samples = 3

    spect_type = sa.form_spectrogram_type(mel, n_fft, log)

    root = './data/train/' + spect_type + '/spect'

    files = glob(os.path.join(root, "*"))
    files = [f for f in files if target in f]

    i = 0
    plt.style.use("dark_background")
    for f in files:
        if i > number_of_samples:
            break
        arr = np.load(f)
        if not log:
            arr = np.log2(arr)
        plt.imshow(arr)
        plt.colorbar()
        plt.show()
        i += 1
        plt.savefig('image_offload/'+target+str(i)+'.png')
        plt.close()
        print(str(i), 'saved to offload directory.')
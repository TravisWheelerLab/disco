import os
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

files = glob("/Users/wheelerlab/beetles_testing/snr_0_test_dat/*pkl")

for f in files:
    with open(f, "rb") as src:
        im = pickle.load(src)

    fig, ax = plt.subplots(nrows=2)
    ax[0].imshow(np.log2(im[0]))
    ax[1].imshow(im[1][np.newaxis, :], vmin=0, vmax=2, aspect="auto")
    ax[0].set_title(os.path.basename(f))
    plt.show()

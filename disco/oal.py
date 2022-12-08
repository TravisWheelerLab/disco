import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

files = glob("/tmp/extracted_test/*")
for f in files:

    with open(f, "rb") as src:
        im = pickle.load(src)

    plt.imshow(np.log2(im[0]))
    plt.title(im[1])
    plt.show()

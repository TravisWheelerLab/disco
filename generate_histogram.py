import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob


def generate_histogram(directory_name, label=''):
    root = os.path.join(directory_name, 'spect')
    files = glob(os.path.join(root, "*"))
    files = [f for f in files if label in f]
    lengths_array = []

    for f in files:
        arr = np.load(f)
        lengths_array.append(arr.shape[1])

    if label == '':
        title = "All class lengths histogram " + directory_name
    else:
        title = "Histogram of " + label + " lengths " + directory_name

    plt.hist(lengths_array, bins=50)
    plt.title(title)
    plt.savefig('image_offload/' + "hist_" + label + '_' + directory_name + '.png')


if __name__ == '__main__':
    generate_histogram('validation_data')
    print("histogram sent to image_offload directory.")

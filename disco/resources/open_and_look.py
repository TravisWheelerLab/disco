import matplotlib.pyplot as plt
import os
import numpy as np
import pdb
import pickle
from glob import glob

def plot_all(target):
    extracted_files = glob(f"/home/tc229954/extracted_data_1150_nffts/{target}/*pkl")
    index_to_color = {0: "r", 2: "#808080", 1: "tab:purple"}
    os.makedirs(f"/home/tc229954/debugging_images/{target}", exist_ok=True)
    for f in extracted_files:
        with open(f, 'rb') as src:
            data = pickle.load(src)

        fig, ax = plt.subplots()
        ax.imshow(np.log2(data[0]))
        for i, label in enumerate(data[1]):
            ax.scatter(i, 0, c=index_to_color[label])

        plt.savefig(f"/home/tc229954/debugging_images/{target}/{os.path.basename(os.path.splitext(f)[0])}.png",
                    bbox_inches="tight")
        plt.close()

def plot_histogram_pointwise(target):

    extracted_files = glob(f"/home/tc229954/extracted_data_1150_nffts/{target}/*pkl")
    os.makedirs(f"/home/tc229954/debugging_images/pointwise_hists/", exist_ok=True)
    all_data = []
    for f in extracted_files:
        with open(f, 'rb') as src:
            data = pickle.load(src)
        all_data.extend(data[1])

    fig, ax = plt.subplots()
    ax.hist(all_data, bins=3)
    plt.savefig(f"/home/tc229954/debugging_images/pointwise_hists/{target}_hist.png")
    plt.close()


plot_histogram_pointwise("test")
plot_histogram_pointwise("train")
plot_histogram_pointwise("validation")

# func("test")
# func("train")
# func("validation")



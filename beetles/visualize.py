import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider
import beetles.inference_utils as infer
from glob import glob
from argparse import ArgumentParser

import beetles.heuristics as heuristics
from beetles.inference_utils import smooth_predictions_with_hmm


def load_arrays(data_root):
    medians = infer.load_pickle(os.path.join(data_root, "median_predictions.pkl"))
    spectrogram = infer.load_pickle(os.path.join(data_root, "raw_spectrogram.pkl"))
    post_hmm = infer.load_pickle(os.path.join(data_root, "hmm_predictions.pkl"))
    iqr = infer.load_pickle(os.path.join(data_root, "iqrs.pkl"))
    return medians, spectrogram, post_hmm, iqr


def main(config, data_path, hop_length, sample_rate):
    # TODO: refactor so this function isn't so massive
    medians, spectrogram, post_hmm, iqr = load_arrays(data_path)
    median_argmax = np.argmax(medians, axis=0)
    medians = medians.T[np.arange(medians.shape[-1]), median_argmax]
    # when the prediction is confidently background (i.e. the argmax
    # results in the background class, set the probabilities to 0.
    medians[median_argmax == config.name_to_class_code["BACKGROUND"]] = 0
    medians = np.expand_dims(medians, axis=0)

    iqr = np.sum(iqr.T, axis=1) / 3
    iqr = np.expand_dims(iqr, axis=0)

    post_hmm[post_hmm != config.name_to_class_code["BACKGROUND"]] = 1
    post_hmm[post_hmm != 1] = 0
    post_hmm = np.expand_dims(post_hmm, axis=0)

    # should be black where the model predicted any class; and white otherwise.
    median_argmax[median_argmax != config.name_to_class_code["BACKGROUND"]] = 1
    median_argmax[median_argmax != 1] = 0
    median_argmax = np.expand_dims(median_argmax, axis=0)

    # prediction_array = np.concatenate((medians, iqr, median_argmax, post_hmm), axis=0)
    prediction_array = np.concatenate((post_hmm, median_argmax, iqr, medians), axis=0)

    prediction_df = infer.load_prediction_csv(
        os.path.join(data_path, "classifications.csv"),
        hop_length=hop_length,
        sample_rate=sample_rate,
    )

    # remove background from prediction df
    prediction_df = prediction_df.loc[prediction_df["Sound_Type"] != "BACKGROUND", :]

    fig, ax = plt.subplots(sharex=True, nrows=2, figsize=(10, 7))

    medians, spectrogram, post_hmm, iqr = load_arrays(data_path)
    iqr_no_mods = iqr.copy()
    median_argmax = np.argmax(medians, axis=0)

    for class_index, name in config.class_code_to_name.items():
        all_class = median_argmax == class_index
        x = range(0, all_class.shape[-1])
        ax[1].fill_between(
            x, 15, 19, where=all_class, color=config.name_to_rgb_code[name]
        )

    post_hmm = infer.smooth_predictions_with_hmm(median_argmax, config=config)
    post_hmm = heuristics.remove_a_chirps_in_between_b_chirps(
        post_hmm, iqr_no_mods, config.name_to_class_code
    )

    for class_index, name in config.class_code_to_name.items():
        all_class = post_hmm == class_index
        x = range(0, all_class.shape[-1])
        ax[1].fill_between(
            x, 10, 14, where=all_class, color=config.name_to_rgb_code[name]
        )

    ax[1].axis("off")
    spectrogram = np.flip(spectrogram, axis=0)

    ax[0].imshow(spectrogram, aspect="auto", origin="lower")
    ax[0].set_ylim([0, spectrogram.shape[0]])

    ax[0].set_title("Raw spectrogram")
    ax[0].set_ylabel("frequency bin")
    ax[0].set_yticks([])
    ax[0].set_xticks([])

    plt.subplots_adjust()
    n = 1200
    ax[1].axis([0, n, 4, 20])
    ax[0].axis([0, n, 0, spectrogram.shape[0]])
    p = ax[0].get_position()
    ax[1].set_position([p.x0, p.y0 - 0.1, p.x1 - p.x0, 0.09])
    fig.text(p.x0 - 0.08, p.y0 - 0.03, "ensemble prediction", fontsize=8)
    fig.text(p.x0 - 0.08, p.y0 - 0.06, "post processed", fontsize=8)

    axpos = plt.axes([p.x0, p.y0 - 0.2, p.x1 - p.x0, 0.05])
    spos = Slider(axpos, "x-position", 0.0, medians.shape[1])

    def update(val):
        ax[1].axis([spos.val, spos.val + n, 4, 20])
        ax[0].axis([spos.val, spos.val + n, 0, spectrogram.shape[0]])

    spos.on_changed(update)
    plt.show()

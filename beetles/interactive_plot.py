import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import beetles.inference_utils as infer
from glob import glob
from argparse import ArgumentParser

import beetles.heuristics
from beetles.inference_utils import smooth_predictions_with_hmm


def main(args):
    # TODO: refactor so this function isn't so massive
    data_root = args.debug_data_path
    hop_length = args.hop_length
    sample_rate = args.sample_rate

    medians = infer.load_pickle(os.path.join(data_root, "median_predictions.pkl"))
    medians = np.expand_dims(np.transpose(medians), 0)
    spectrogram = infer.load_pickle(os.path.join(data_root, "raw_spectrogram.pkl"))

    hmm_predictions = infer.convert_argmaxed_array_to_rgb(
        infer.load_pickle(os.path.join(data_root, "hmm_predictions.pkl"))
    )

    prediction_df = infer.load_prediction_csv(
        os.path.join(data_root, "classifications.csv"),
        hop_length=hop_length,
        sample_rate=sample_rate,
    )

    # remove prediction df
    prediction_df = prediction_df.loc[prediction_df["Sound_Type"] != "BACKGROUND", :]

    fig, ax = plt.subplots(sharex=True, nrows=2, figsize=(10, 7))

    iqr = infer.load_pickle(os.path.join(data_root, "iqrs.pkl"))

    predictions = np.argmax(medians.squeeze(), axis=1)

    predictions_rgb = infer.convert_argmaxed_array_to_rgb(predictions)
    iqr = np.expand_dims(np.transpose(iqr), 0)

    prediction_array = np.concatenate(
        (hmm_predictions, predictions_rgb, iqr, medians), axis=0
    )

    ax[0].imshow(spectrogram, aspect="auto")
    ax[0].set_ylim([0, spectrogram.shape[0]])

    # TODO: refactor so I only plot in a window
    for class_index, name in infer.CLASS_CODE_TO_NAME.items():
        subdf = prediction_df.loc[prediction_df["Sound_Type"] == name, :]
        ax[0].vlines(
            subdf["Begin Spect Index"],
            ymin=0,
            ymax=spectrogram.shape[0],
            colors=infer.SOUND_TYPE_TO_COLOR[name],
        )
    for class_index, name in infer.CLASS_CODE_TO_NAME.items():
        subdf = prediction_df.loc[prediction_df["Sound_Type"] == name, :]
        ax[0].vlines(
            subdf["End Spect Index"],
            ymin=0,
            ymax=spectrogram.shape[0],
            linestyles="dashed",
            colors=infer.SOUND_TYPE_TO_COLOR[name],
        )

    for cls in infer.SOUND_TYPE_TO_COLOR.keys():
        if cls != "BACKGROUND":
            ax[0].plot(
                [0, 0],
                [0, spectrogram.shape[0]],
                "{}-".format(infer.SOUND_TYPE_TO_COLOR[cls]),
                label="begin of {} chirp".format(cls),
            )
    for cls in infer.SOUND_TYPE_TO_COLOR.keys():
        if cls != "BACKGROUND":
            ax[0].plot(
                [0, 0],
                [0, spectrogram.shape[0]],
                "{}-.".format(infer.SOUND_TYPE_TO_COLOR[cls]),
                label="end of {} chirp".format(cls),
            )

    ax[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax[1].imshow(prediction_array, aspect="auto", interpolation="nearest")
    ax[1].set_yticks([0, 1, 2, 3])
    ax[1].set_xticks([])
    ax[1].set_xlabel("spectrogram record")
    ax[1].set_yticklabels(
        ["heuristics + smoothing w/ hmm", "median argmax", "iqr", "median predictions"],
        rotation=45,
    )
    ax[1].set_title(
        "Predictions mapped to RGB values. red: A chirp, green: B chirp, blue: background"
    )
    ax[0].set_title("Raw spectrogram")
    ax[0].set_ylabel("frequency bin")

    plt.subplots_adjust(right=0.7)
    axcolor = "lightgoldenrodyellow"
    axpos = plt.axes([0.2, 0.0001, 0.65, 0.03], facecolor=axcolor)
    spos = Slider(axpos, "x-position", 0.0, medians.shape[1])

    n = 800
    ax[1].axis([0, n, -0.5, 3.5])
    ax[0].axis([0, n, 0, spectrogram.shape[0]])

    def update(val):
        pos = spos.val
        ax[1].axis([pos, pos + n, -0.5, 3.5])
        ax[0].axis([pos, pos + n, 0, spectrogram.shape[0]])
        fig.canvas.draw_idle()

    spos.on_changed(update)
    plt.show()

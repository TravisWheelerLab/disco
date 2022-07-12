import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.widgets import Slider
import disco.inference_utils as infer


class Visualizer:
    def __init__(self, data_path, medians, post_process, means, iqr, config):
        self.config = config
        self.spectrogram, self.medians, self.post_hmm, self.iqr, self.means, self.votes = load_arrays(data_path)

        self.spectrogram = np.flip(self.spectrogram, axis=0)
        self.median_argmax = np.argmax(self.medians, axis=0)
        self.mean_argmax = np.argmax(self.means, axis=0)

        self.statistics = []
        self.show_legend = False
        if medians:
            self.statistics.append(("ensemble preds (medians)", self.median_argmax))
        if post_process:
            self.statistics.append(("post process (medians)", self.post_hmm))
        if means:
            self.statistics.append(("ensemble preds (means)", self.mean_argmax))
        if iqr:
            self.iqr = np.mean(self.iqr, axis=0)
            self.statistics.append(("ensemble iqr (medians)", self.iqr))


def load_arrays(data_root):
    medians = infer.load_pickle(os.path.join(data_root, "median_predictions.pkl"))
    spectrogram = infer.load_pickle(os.path.join(data_root, "raw_spectrogram.pkl"))
    post_hmm = infer.load_pickle(os.path.join(data_root, "hmm_predictions.pkl"))
    iqr = infer.load_pickle(os.path.join(data_root, "iqrs.pkl"))
    means = infer.load_pickle(os.path.join(data_root, "mean_predictions.pkl"))
    votes = infer.load_pickle(os.path.join(data_root, "votes.pkl"))
    return spectrogram, medians, post_hmm, iqr, means, votes


def add_predictions_legend(ax, config):
    legend_handles = []
    for name in config.name_to_rgb_code.keys():
        icon = mlines.Line2D([], [], color=config.name_to_rgb_code[name], marker="s", linestyle='None', markersize=10,
                             label=name.title())
        legend_handles.append(icon)
    ax.legend(handles=legend_handles, loc='upper right', fontsize='small', title='prediction')


def visualize(config, data_path, medians, post_process, means, iqr):
    """
    Visualize predictions interactively.
    :param config: disco.Config() object.
    :param data_path: path of directory containing spectrogram and inference ran on it.
    :param medians: whether to display median predictions by the ensemble.
    :param post_process: whether to display post-processed (hmm, other heuristics) predictions by the ensemble.
    :param means: whether to display mean predictions by the ensemble.
    :return:
    """
    visualizer = Visualizer(data_path, medians, post_process, means, iqr, config)

    # Build the figure height based on the height of the spectrogram and the amount of statistics we want to display.
    num_statistics_displayed = len(visualizer.statistics)
    spect_ht = 2.5
    statistics_ht = (0.5 + (num_statistics_displayed - 1) + (num_statistics_displayed - 2)*0.1) * 0.22
    fig_ht = spect_ht + statistics_ht

    # Create width ratios so the spectrogram's window is bigger than the statistics below it, and the
    #   statistics all have the same size.
    statistics_display_sizes = np.repeat(statistics_ht/num_statistics_displayed, num_statistics_displayed).tolist()
    subplot_sizes = [spect_ht] + statistics_display_sizes
    height_ratios = {'height_ratios': subplot_sizes}

    fig, axs = plt.subplots(sharex=True, nrows=num_statistics_displayed+1,
                            figsize=(10, fig_ht), gridspec_kw=height_ratios)
    fig.subplots_adjust(top=1 - 0.35 / fig_ht, bottom=0.15 / fig_ht, left=0.2, right=0.99)

    # Show spectrogram
    axs[0].imshow(visualizer.spectrogram, aspect="auto")

    for i in range(1, len(axs)):
        print(i)

    # Show each statistics row
    for i in range(1, len(axs)):
        label = visualizer.statistics[i-1][0]
        statistics_bar = np.expand_dims(visualizer.statistics[i-1][1], axis=0)
        if label == "ensemble preds (medians)" or label == "post process (medians)" or label == "ensemble preds (means)":
            # todo: make colormap discrete, map values to colors
            cmap = "gist_rainbow"
        elif label == "ensemble iqr (medians)":
            cmap = "plasma"
        axs[i].imshow(statistics_bar, aspect="auto", cmap=cmap)
        axs[i].text(-0.01, 0.5, label, va="center", ha="right", fontsize=10, transform=axs[i].transAxes)

    # turn off tick marks for each statistics bar
    for i in range(1, len(axs)):
        axs[i].set_axis_off()

    if visualizer.show_legend:
        add_predictions_legend(axs[0], config)

    plt.show()

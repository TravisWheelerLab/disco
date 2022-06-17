import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.widgets import Slider
import disco.inference_utils as infer


class Visualizer:
    def __init__(self, data_path, medians, post_process, config):
        self.config = config
        self.spectrogram, self.medians, self.post_hmm, self.iqr, self.means, self.votes = load_arrays(data_path)

        self.spectrogram = np.flip(self.spectrogram, axis=0)
        self.median_argmax = np.argmax(self.medians, axis=0)
        self.mean_argmax = np.argmax(self.means, axis=0)

        self.displayed_statistics = []
        if medians:
            self.displayed_statistics.append(self.median_argmax)
        if post_process:
            self.displayed_statistics.append(self.post_hmm)


def load_arrays(data_root):
    medians = infer.load_pickle(os.path.join(data_root, "median_predictions.pkl"))
    spectrogram = infer.load_pickle(os.path.join(data_root, "raw_spectrogram.pkl"))
    post_hmm = infer.load_pickle(os.path.join(data_root, "hmm_predictions.pkl"))
    iqr = infer.load_pickle(os.path.join(data_root, "iqrs.pkl"))
    means = infer.load_pickle(os.path.join(data_root, "mean_predictions.pkl"))
    votes = infer.load_pickle(os.path.join(data_root, "votes.pkl"))
    return spectrogram, medians, post_hmm, iqr, means, votes


def add_statistical_visualizations(visualizer, ax):
    for statistic_index in range(len(visualizer.displayed_statistics)):
        add_predictions_bar(visualizer.median_argmax, ax, 15-5*statistic_index, 19-5*statistic_index, visualizer.config)


def add_predictions_bar(output, ax, y1, y2, config):
    for class_index, name in config.class_code_to_name.items():
        all_class = output == class_index
        x = range(0, all_class.shape[-1])
        ax[1].fill_between(x, y1, y2, where=all_class, color=config.name_to_rgb_code[name])


def set_up_spectrogram_axes(spectrogram, ax):
    ax[0].imshow(spectrogram, aspect="auto", origin="lower")
    ax[0].set_ylim([0, spectrogram.shape[0]])

    ax[0].set_title("raw spectrogram")
    ax[0].set_ylabel("frequency bin")
    ax[0].set_yticks([])
    ax[0].set_xticks([])


def set_up_figure_positioning(ax, visualizer):
    ax[1].axis([0, visualizer.config.visualization_zoom_out, 4, 20])
    ax[0].axis([0, visualizer.config.visualization_zoom_out, 0, visualizer.spectrogram.shape[0]])
    spect_position = ax[0].get_position()
    ax[1].set_position([spect_position.x0, spect_position.y0 - 0.1, spect_position.x1 - spect_position.x0, 0.09])


def add_prediction_bar_labels(fig, spect_position):
    fig.text(spect_position.x0 - 0.08, spect_position.y0 - 0.03, "ensemble pred (medians)", fontsize=8)
    fig.text(spect_position.x0 - 0.08, spect_position.y0 - 0.06, "post process (medians)", fontsize=8)


def add_predictions_legend(ax, config):
    background_icon = mlines.Line2D([], [], color="#AAAAAA", marker="s", linestyle='None', markersize=10,
                                    label="Background")
    a_icon = mlines.Line2D([], [], color="#b65b47", marker="s", linestyle='None', markersize=10, label="A Chirp")
    b_icon = mlines.Line2D([], [], color="#A36DE9", marker="s", linestyle='None', markersize=10, label="B Chirp")

    legend_handles = [background_icon, a_icon, b_icon]
    ax[0].legend(handles=legend_handles, loc='upper right', fontsize='small', title='prediction')


def visualize(config, data_path, medians, post_process):
    """
    Visualize predictions interactively.
    :param config: disco.Config() object.
    :param data_path: path of directory containing spectrogram and inference ran on it.
    :param medians: whether to display median predictions by the ensemble.
    :param post_process: whether to display post-processed (hmm, other heuristics) predictions by the ensemble.
    :param means: whether to display mean predictions by the ensemble.
    :return:
    """
    fig, ax = plt.subplots(sharex=True, nrows=2, figsize=(10, 7))
    visualizer = Visualizer(data_path, medians, post_process, config)

    add_statistical_visualizations(visualizer, ax)

    ax[1].axis("off")

    set_up_spectrogram_axes(visualizer.spectrogram, ax)

    plt.subplots_adjust()

    set_up_figure_positioning(ax, visualizer)

    spect_position = ax[0].get_position()
    add_prediction_bar_labels(fig, spect_position)
    add_predictions_legend(ax, config)

    axis_position = plt.axes([spect_position.x0, spect_position.y0-0.2, spect_position.x1 - spect_position.x0, 0.05])

    slider = Slider(axis_position, "x-position", 0.0, visualizer.medians.shape[1])

    def update(val):
        ax[1].axis([slider.val, slider.val + config.visualization_zoom_out, 4, 20])
        ax[0].axis([slider.val, slider.val + config.visualization_zoom_out, 0, visualizer.spectrogram.shape[0]])

    slider.on_changed(update)
    plt.show()

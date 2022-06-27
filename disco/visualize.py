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

        self.displayed_statistics = []
        if medians:
            self.displayed_statistics.append(self.median_argmax)
        if post_process:
            self.displayed_statistics.append(self.post_hmm)
        if means:
            self.displayed_statistics.append(self.mean_argmax)
        if iqr:
            self.displayed_statistics.append(self.iqr)

        self.statistics_dict = {"ensemble preds (medians)": [medians, self.median_argmax],
                                "post process (medians)": [post_process, self.post_hmm],
                                "ensemble preds (means)": [means, self.mean_argmax],
                                "ensemble iqr (medians)": [iqr, self.iqr]}


def load_arrays(data_root):
    medians = infer.load_pickle(os.path.join(data_root, "median_predictions.pkl"))
    spectrogram = infer.load_pickle(os.path.join(data_root, "raw_spectrogram.pkl"))
    post_hmm = infer.load_pickle(os.path.join(data_root, "hmm_predictions.pkl"))
    iqr = infer.load_pickle(os.path.join(data_root, "iqrs.pkl"))
    means = infer.load_pickle(os.path.join(data_root, "mean_predictions.pkl"))
    votes = infer.load_pickle(os.path.join(data_root, "votes.pkl"))
    return spectrogram, medians, post_hmm, iqr, means, votes


def add_statistical_visualizations(visualizer, ax):
    key_index = 0
    for key in visualizer.statistics_dict.keys():
        if key == "ensemble preds (medians)" or key == "post process (medians)" or key == "ensemble preds (means)":
            if visualizer.statistics_dict[key][0]:
                add_predictions_bar(visualizer.statistics_dict[key][1], ax, 15 - 5*key_index,
                                    19 - 5*key_index, visualizer.config)
                key_index += 1
        elif key == "ensemble iqr (medians)":
            if visualizer.statistics_dict[key][0]:
                add_iqr_bar(visualizer.statistics_dict[key][1], ax, 15 - 5*key_index, 19 - 5*key_index)


def add_predictions_bar(output_array, ax, y1, y2, config):
    for class_index, name in config.class_code_to_name.items():
        all_class = output_array == class_index
        x = range(0, all_class.shape[-1])
        ax[1].fill_between(x, y1, y2, where=all_class, color=config.name_to_rgb_code[name])


def add_iqr_bar(output_array, ax, y1, y2):
    # todo: debug this so it shows a colormap rather than a solid blue
    average_across_classes = np.mean(output_array, axis=0)
    x = range(0, average_across_classes.shape[-1])
    ax[1].fill_between(x, y1, y2, where=average_across_classes, cmap="viridis")


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


def add_prediction_bar_labels(fig, spect_position, statistics_dict):
    index_iterator = 0.03
    for key in statistics_dict.keys():
        if statistics_dict[key][0]:
            fig.text(spect_position.x0 - 0.08, spect_position.y0 - index_iterator, key, fontsize=8)
            index_iterator += 0.03


def add_predictions_legend(ax, config):
    legend_handles = []
    for name in config.name_to_rgb_code.keys():
        icon = mlines.Line2D([], [], color=config.name_to_rgb_code[name], marker="s", linestyle='None', markersize=10,
                             label=name.title())
        legend_handles.append(icon)
    ax[0].legend(handles=legend_handles, loc='upper right', fontsize='small', title='prediction')


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
    fig, ax = plt.subplots(sharex=True, nrows=2, figsize=(10, 7))
    visualizer = Visualizer(data_path, medians, post_process, means, iqr, config)

    add_statistical_visualizations(visualizer, ax)

    ax[1].axis("off")

    set_up_spectrogram_axes(visualizer.spectrogram, ax)

    plt.subplots_adjust()

    set_up_figure_positioning(ax, visualizer)

    spect_position = ax[0].get_position()
    add_prediction_bar_labels(fig, spect_position, visualizer.statistics_dict)

    if medians or post_process or means:
        add_predictions_legend(ax, config)

    axis_position = plt.axes([spect_position.x0, spect_position.y0 - 0.2, spect_position.x1 - spect_position.x0, 0.05])

    slider = Slider(axis_position, "x-position", 0.0, visualizer.medians.shape[1])

    def update(val):
        ax[1].axis([slider.val, slider.val + config.visualization_zoom_out, 4, 20])
        ax[0].axis([slider.val, slider.val + config.visualization_zoom_out, 0, visualizer.spectrogram.shape[0]])

    slider.on_changed(update)
    plt.show()

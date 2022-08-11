import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.widgets import Slider
from matplotlib.colors import to_rgb
import disco.inference_utils as infer


# todo: Instead of passing in just config, give functions the specific things config uses for better functionality
#  elsewhere.
# todo: put argmax into create_statistics_array function


class Visualizer:
    def __init__(self, data_path, medians, post_process, means, iqr, votes, votes_line, second_data_path, config):
        self.config = config
        self.votes_line = votes_line
        self.spectrogram, self.medians, self.post_hmm, self.iqr, self.means, self.votes = load_arrays(data_path)
        self.spectrogram = np.flip(self.spectrogram, axis=0)

        first_data_path_name = os.path.split(data_path)[-1].split("-")[1].split("_")[0]

        self.median_argmax = np.argmax(self.medians, axis=0)
        self.mean_argmax = np.argmax(self.means, axis=0)
        self.statistics, self.show_legend = create_statistics_array(medians, self.median_argmax,
                                                                    post_process, self.post_hmm,
                                                                    means, self.mean_argmax,
                                                                    iqr, self.iqr,
                                                                    votes, self.votes,
                                                                    config.class_code_to_name,
                                                                    first_data_path_name + ":")

        if second_data_path:
            _, self.medians_2, self.post_hmm_2, self.iqr_2, self.means_2, self.votes_2 = load_arrays(second_data_path)
            second_data_path_name = os.path.split(second_data_path)[-1].split("-")[1].split("_")[0]
            self.median_argmax_2 = np.argmax(self.medians_2, axis=0)
            self.mean_argmax_2 = np.argmax(self.means_2, axis=0)
            # todo: make sure _ and self.spectrogram are the same, throw an error if they are not.
            self.statistics_2, _ = create_statistics_array(medians, self.median_argmax_2,
                                                           post_process, self.post_hmm_2,
                                                           means, self.mean_argmax_2,
                                                           iqr, self.iqr_2,
                                                           votes, self.votes_2,
                                                           config.class_code_to_name,
                                                           second_data_path_name + ":")
            self.statistics += self.statistics_2

        # Build the figure height based on the height of the spectrogram and the amount of statistics to display.
        self.num_statistics_plus_slider = len(self.statistics) + 1
        self.spect_ht = 2.5
        statistics_ht = (0.5 + (self.num_statistics_plus_slider-1) + (self.num_statistics_plus_slider-2) * 0.1) * 0.22
        self.fig_ht = self.spect_ht + statistics_ht

        self.height_ratios = get_subplot_ht_ratios(statistics_ht, self.num_statistics_plus_slider, self.spect_ht)


def load_arrays(data_root):
    medians = infer.load_pickle(os.path.join(data_root, "median_predictions.pkl"))
    spectrogram = infer.load_pickle(os.path.join(data_root, "raw_spectrogram.pkl"))
    post_hmm = infer.load_pickle(os.path.join(data_root, "hmm_predictions.pkl"))
    iqr = infer.load_pickle(os.path.join(data_root, "iqrs.pkl"))
    means = infer.load_pickle(os.path.join(data_root, "mean_predictions.pkl"))
    votes = infer.load_pickle(os.path.join(data_root, "votes.pkl"))
    return spectrogram, medians, post_hmm, iqr, means, votes


def create_statistics_array(show_medians, median_argmax, show_post_process, post_hmm, show_means, mean_argmax, show_iqr,
                            iqr, show_votes, votes, class_code_to_name, dataset_name):
    # todo: make it so dataset name is only shown when there is a second dataset
    statistics = []
    show_legend = False
    if show_medians:
        statistics.append((dataset_name + " ensemble preds (medians)", median_argmax))
        show_legend = True
    if show_post_process:
        statistics.append((dataset_name + " post process (medians)", post_hmm))
        show_legend = True
    if show_means:
        statistics.append((dataset_name + " ensemble preds (means)", mean_argmax))
        show_legend = True
    if show_iqr:
        iqr = np.mean(iqr, axis=0)
        statistics.append((dataset_name + " ensemble iqr (medians)", iqr))
    if show_votes:
        for class_code in range(votes.shape[0]):
            text = dataset_name + " votes for " + class_code_to_name[class_code]
            statistics.append((text, votes[class_code, :]))
    return statistics, show_legend


def get_subplot_ht_ratios(height_of_statistics_portion, num_statistics_plus_slider, spectrogram_height):
    # Create width ratios so the spectrogram's window is bigger than the statistics below it, and the
    #   statistics all have the same size.
    statistics_display_sizes = np.repeat(height_of_statistics_portion / num_statistics_plus_slider,
                                         num_statistics_plus_slider).tolist()
    subplot_sizes = [spectrogram_height] + statistics_display_sizes
    height_ratios = {'height_ratios': subplot_sizes}
    return height_ratios


def imshow_statistics_rows(axs, visualizer, config):
    # Show each statistics row
    for i in range(1, len(axs) - 1):
        label = visualizer.statistics[i - 1][0]
        statistics_bar = np.expand_dims(visualizer.statistics[i - 1][1], axis=0)
        if "preds" in label or "post process" in label:
            color_dict = dict()
            for class_code in range(len(config.class_code_to_name.keys())):
                class_hex_code = config.name_to_rgb_code[config.class_code_to_name[class_code]]
                class_rgb_code = np.array(to_rgb(class_hex_code))
                color_dict[class_code] = class_rgb_code
            statistics_rgb = np.expand_dims(np.array([color_dict[i] for i in np.squeeze(statistics_bar)]), axis=0)
            axs[i].imshow(statistics_rgb, aspect="auto")
        else:
            if "iqr" in label:
                x = np.arange(start=0, stop=statistics_bar.shape[-1])
                y = visualizer.statistics[i - 1][1]
                axs[i].scatter(x, y, s=0.25, color="#000000")
                axs[i].set_ylim([0, 1])
            elif "votes for" in label:
                if visualizer.votes_line:
                    x = np.arange(start=0, stop=statistics_bar.shape[-1])
                    y = visualizer.statistics[i - 1][1]
                    axs[i].plot(x, y, color="#000000")
                    axs[i].set_ylim([0, 10])
                else:
                    cmap = "Blues"
                    axs[i].imshow(statistics_bar, aspect="auto", cmap=cmap)
        axs[i].text(-0.01, 0.5, label, va="center", ha="right", fontsize=10, transform=axs[i].transAxes)

    # turn off tick marks for each statistics bar and for the slider bar
    for i in range(1, len(axs)):
        slider_axs_idx = len(axs)-1
        if i != slider_axs_idx and "iqr" not in visualizer.statistics[i-1][0]:
            axs[i].set_axis_off()
        elif i == slider_axs_idx:
            axs[i].get_yaxis().set_visible(False)


def add_predictions_legend(ax, config):
    legend_handles = []
    for name in config.name_to_rgb_code.keys():
        icon = mlines.Line2D([], [], color=config.name_to_rgb_code[name], marker="s", linestyle='None', markersize=10,
                             label=name.title())
        legend_handles.append(icon)
    ax.legend(handles=legend_handles, loc='upper right', fontsize='small', title='prediction')


def build_slider(axs, visualizer):
    # todo: rename to something other than spect_position
    spect_position = axs[len(visualizer.statistics) + 1].get_position()
    axis_position = plt.axes([spect_position.x0, spect_position.y0, spect_position.x1 - spect_position.x0, 0.05])
    slider = Slider(axis_position, "x-position", 0.0, visualizer.statistics[0][1].shape[0])
    return slider


def visualize(config, data_path, medians, post_process, means, iqr, votes, votes_line, second_data_path):
    """
    Visualize predictions interactively.
    :param config: disco.Config() object.
    :param data_path: path of directory containing spectrogram and inference ran on it.
    :param medians: whether to display median predictions by the ensemble.
    :param post_process: whether to display post-processed (hmm, other heuristics) predictions by the ensemble.
    :param means: whether to display mean predictions by the ensemble.
    :return:
    """
    visualizer = Visualizer(data_path, medians, post_process, means, iqr, votes, votes_line, second_data_path, config)

    fig, axs = plt.subplots(sharex=True, nrows=visualizer.num_statistics_plus_slider + 1,
                            figsize=(10, visualizer.fig_ht), gridspec_kw=visualizer.height_ratios)
    fig.subplots_adjust(top=1 - 0.35 / visualizer.fig_ht, bottom=0.15 / visualizer.fig_ht, left=0.2, right=0.99)

    axs[0].imshow(visualizer.spectrogram, aspect="auto")
    imshow_statistics_rows(axs, visualizer, config)

    if visualizer.show_legend:
        add_predictions_legend(axs[0], config)

    slider = build_slider(axs, visualizer)

    def update(val):
        for i in range(len(axs) - 1):
            axs[i].set_xlim(slider.val, slider.val + config.visualization_zoom_out)

    slider.on_changed(update)

    plt.show()

import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.widgets import Slider

import disco_sound.util.inference_utils as infer


class Visualizer:
    def __init__(
        self,
        data_path,
        medians,
        post_process,
        means,
        iqr,
        votes,
        votes_line,
        second_data_path,
        class_code_to_name,
    ):
        """
        Class containing most information needed to perform visualization.
        :param data_path: String. data path to the directory containing the csv of predictions and pkls of statistics.
        :param medians: bool. Whether to display an array containing the class with the highest median softmax value
        for each spectrogram index. The median for each time point is found for each class across the entire ensemble
        during inference.
        :param post_process: bool. Whether to display post-HMM and other heuristic-changed predictions.
        :param means: bool. Whether to display means (see medians for how this is calculated).
        :param iqr: bool. Whether to display average iqr across each class for each time point
        :param votes: bool. Whether to display heatmap of votes for each class.
        :param votes_line: bool. Whether to display votes as a line rather than a color bar.
        :param second_data_path: String. data path to visualizations from a different ensemble of the same .wav file.
        Useful for comparing multiple ensembles and determining which is better.
        :return: None.
        """

        self.votes_line = votes_line
        (
            self.spectrogram,
            self.median_argmax,
            self.post_hmm,
            self.iqr,
            self.means,
            self.votes,
        ) = load_arrays(data_path)
        self.spectrogram = np.flip(self.spectrogram, axis=0)

        if self.median_argmax.max() > max(class_code_to_name.keys()):
            self.class_code_to_name[
                self.median_argmax.max()
            ] = "DROPPED, HIGH UNCERTAINTY"
            self.name_to_rgb_code["DROPPED, HIGH UNCERTAINTY"] = "#282B30"

        first_data_path_name = os.path.split(data_path)[-1].split("-")[-1].split("_")[0]

        self.mean_argmax = np.argmax(self.means, axis=0)
        self.statistics, self.show_legend = create_statistics_array(
            medians,
            self.median_argmax,
            post_process,
            self.post_hmm,
            means,
            self.mean_argmax,
            iqr,
            self.iqr,
            votes,
            self.votes,
            class_code_to_name,
            first_data_path_name + ":",
        )
        if second_data_path:
            (
                _,
                self.median_argmax_2,
                self.post_hmm_2,
                self.iqr_2,
                self.means_2,
                self.votes_2,
            ) = load_arrays(second_data_path)
            second_data_path_name = (
                os.path.split(second_data_path)[-1].split("-")[1].split("_")[0]
            )
            self.mean_argmax_2 = np.argmax(self.means_2, axis=0)
            # todo: make sure _ and self.spectrogram are the same, throw an error if they are not.
            self.statistics_2, _ = create_statistics_array(
                medians,
                self.median_argmax_2,
                post_process,
                self.post_hmm_2,
                means,
                self.mean_argmax_2,
                iqr,
                self.iqr_2,
                votes,
                self.votes_2,
                class_code_to_name,
                second_data_path_name + ":",
            )
            self.statistics += self.statistics_2

        # Build the figure height based on the height of the spectrogram and the amount of statistics to display.
        self.num_statistics_plus_slider = len(self.statistics) + 1
        self.spect_ht = 2.5
        statistics_ht = (
            0.5
            + (self.num_statistics_plus_slider - 1)
            + (self.num_statistics_plus_slider - 2) * 0.1
        ) * 0.22
        self.fig_ht = self.spect_ht + statistics_ht

        self.height_ratios = get_subplot_ht_ratios(
            statistics_ht, self.num_statistics_plus_slider, self.spect_ht
        )


def load_arrays(data_root):
    """
    Get numpy arrays of each statistic saved in the provided visualization data directory.
    :param data_root: String of the user-provided visualization data directory.
    """
    medians = infer.load_pickle(os.path.join(data_root, "median_predictions.pkl"))
    if os.path.isfile(os.path.join(data_root, "raw_spectrogram.pkl")):
        spectrogram = infer.load_pickle(os.path.join(data_root, "raw_spectrogram.pkl"))
    else:
        spectrogram = infer.load_pickle(os.path.join(data_root, "spectrogram.pkl"))

    post_hmm = infer.load_pickle(os.path.join(data_root, "hmm_predictions.pkl"))
    iqr = infer.load_pickle(os.path.join(data_root, "iqrs.pkl"))
    if not os.path.isfile(os.path.join(data_root, "mean_predictions.pkl")):
        means = infer.load_pickle(os.path.join(data_root, "median_predictions.pkl"))
    else:
        means = infer.load_pickle(os.path.join(data_root, "mean_predictions.pkl"))
    votes = infer.load_pickle(os.path.join(data_root, "votes.pkl"))
    return spectrogram, medians, post_hmm, iqr, means, votes


def create_statistics_array(
    show_medians,
    median_argmax,
    show_post_process,
    post_hmm,
    show_means,
    mean_argmax,
    show_iqr,
    iqr,
    show_votes,
    votes,
    class_code_to_name,
    dataset_name,
):
    """
    Get numpy arrays of each statistic saved in the provided visualization data directory.
    :param show_medians: bool. Whether to add the medians-based predictions to the visualizer.
    :param median_argmax: Numpy array of the median-based predictions (calculation explained in Visualizer __init__).
    :param show_post_process: bool. Whether to add the HMM- and other heuristic-smoothed predictions to the visualizer.
    :param post_hmm: Numpy array of the smoothed predictions.
    :param show_means: bool. Whether to add the means-based predictions to the visualizer.
    :param mean_argmax: Numpy array of the mean-based predictions.
    :param show_iqr: bool. Whether to add the average ensemble iqr across classes per timepoint to the visualizer.
    :param iqr: Numpy array of the average iqr.
    :param show_votes: bool. Whether to add the ensemble's votes for each class to the visualizer.
    :param votes: Numpy array of the votes for each class, size: (class, length).
    :param class_code_to_name: Dictionary mapping numpy array values to the actual labels, used for creating the legend.
    :param dataset_name: Simplified version of dataset displayed in the title of each statistic.
    :return: statistics array containing tuples of (statistic name, statistic array); bool. of whether to show the
    legend of class predictions.
    """
    # todo: make it so dataset name is only shown when there is a second dataset
    # todo: make into a class function to eliminate most arguments
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


def get_subplot_ht_ratios(
    height_of_statistics_portion, num_statistics_plus_slider, spectrogram_height
):
    """
    Creates a dictionary that can be ingested by the visualization figure that indicates the size of each display.
    Ensures that the spectrogram's window is bigger than the statistics bars below it and that the statistics
    all display the same size.
    :param height_of_statistics_portion: float indicating relative height of the statistics portion.
    :param num_statistics_plus_slider: int. containing the number of displayed statistics plus 1 (for the slider).
    :param spectrogram_height: float indicating relative height of the spectrogram.
    :return: Dictionary mapping 'height_ratios' to a list containing the proportions of each piece of the display to be
    taken in by the matplotlib figure creation (the pieces are the spectrogram, the statistics, and the slider bar).
    """
    statistics_display_sizes = np.repeat(
        height_of_statistics_portion / num_statistics_plus_slider,
        num_statistics_plus_slider,
    ).tolist()
    subplot_sizes = [spectrogram_height] + statistics_display_sizes
    height_ratios = {"height_ratios": subplot_sizes}
    return height_ratios


def imshow_statistics_rows(
    axs,
    visualizer,
    class_code_to_name,
    name_to_rgb_code,
):
    """
    Go through each subplot (row) and display each statistic.
    :param axs: Matplotlib axes containing each subplot.
    :param visualizer: Visualizer object with all statistics needed for display.
    :return: None.
    """
    for i in range(1, len(axs) - 1):
        label = visualizer.statistics[i - 1][0]
        statistics_bar = np.expand_dims(
            visualizer.statistics[i - 1][1], axis=0
        ).squeeze()
        if "preds" in label or "post process" in label:
            color_dict = dict()
            for class_code in range(len(class_code_to_name.keys())):
                class_hex_code = name_to_rgb_code[class_code_to_name[class_code]]
                class_rgb_code = np.array(to_rgb(class_hex_code))
                color_dict[class_code] = class_rgb_code

            if statistics_bar.ndim > 1:
                statistics_bar = statistics_bar.argmax(axis=0)

            statistics_rgb = np.expand_dims(
                np.array([color_dict[i] for i in np.squeeze(statistics_bar)]), axis=0
            )
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
        axs[i].text(
            -0.01,
            0.5,
            label,
            va="center",
            ha="right",
            fontsize=10,
            transform=axs[i].transAxes,
        )

    # turn off tick marks for each statistics bar and for the slider bar
    for i in range(1, len(axs)):
        slider_axs_idx = len(axs) - 1
        if i != slider_axs_idx and "iqr" not in visualizer.statistics[i - 1][0]:
            axs[i].set_axis_off()
        elif i == slider_axs_idx:
            axs[i].get_yaxis().set_visible(False)


def add_predictions_legend(ax, name_to_rgb_code):
    """
    Creates a legend for class predictions in the top right corner of the spectrogram.
    :param ax: Matplotlib subplot containing the spectrogram
    :return: None.
    """
    legend_handles = []
    for name in name_to_rgb_code.keys():
        icon = mlines.Line2D(
            [],
            [],
            color=name_to_rgb_code[name],
            marker="s",
            linestyle="None",
            markersize=10,
            label=name.title(),
        )
        legend_handles.append(icon)
    ax.legend(
        handles=legend_handles, loc="upper right", fontsize="small", title="prediction"
    )


def build_slider(axs, visualizer):
    """
    Creates a slider used to move across the spectrogram and its visualization display.
    :param axs: All Matplotlib subplots in figure.
    :param visualizer: Visualizer object.
    :return: Matplotlib Slider object with information it needs to initially display.
    """
    # todo: rename to something other than spect_position
    spect_position = axs[len(visualizer.statistics) + 1].get_position()
    axis_position = plt.axes(
        [
            spect_position.x0,
            spect_position.y0,
            spect_position.x1 - spect_position.x0,
            0.05,
        ]
    )
    slider = Slider(
        axis_position, "x-position", valmin=0.0, valmax=visualizer.spectrogram.shape[-1]
    )

    return slider


def visualize(
    data_path,
    medians,
    post_process,
    means,
    iqr,
    votes,
    votes_line,
    second_data_path,
    class_code_to_name,
    name_to_rgb_code,
    visualization_columns,
    seed=None,
):
    """
    Visualize predictions interactively.
    :param data_path: String. path of directory containing spectrogram and inference ran on it.
    :param medians: bool. whether to display median predictions by the ensemble.
    :param post_process: bool. whether to display post-processed (hmm, other heuristics) predictions by the ensemble.
    :param means: bool. whether to display mean predictions by the ensemble.
    :param iqr: bool. whether to display average iqr across classes for each spectrogram index.
    :param votes: bool. whether to display votes for each class for each spectrogram index.
    :param votes_line: bool. whether to display votes with a line rather than a colorbar.
    :param second_data_path: String. filepath of second ensemble's visualization statistics for the same spectrogram.
    :return: None.
    """
    visualizer = Visualizer(
        data_path,
        medians,
        post_process,
        means,
        iqr,
        votes,
        votes_line,
        second_data_path,
        class_code_to_name,
    )

    fig, axs = plt.subplots(
        sharex=True,
        nrows=visualizer.num_statistics_plus_slider + 1,
        figsize=(10, visualizer.fig_ht),
        gridspec_kw=visualizer.height_ratios,
    )
    fig.subplots_adjust(
        top=1 - 0.35 / visualizer.fig_ht,
        bottom=0.15 / visualizer.fig_ht,
        left=0.2,
        right=0.99,
    )

    axs[0].imshow(visualizer.spectrogram, aspect="auto")
    imshow_statistics_rows(axs, visualizer, class_code_to_name, name_to_rgb_code)

    if visualizer.show_legend:
        add_predictions_legend(axs[0], name_to_rgb_code)

    slider = build_slider(axs, visualizer)

    def update(val):
        for i in range(len(axs) - 1):
            axs[i].set_xlim(
                slider.val,
                slider.val + visualization_columns,
            )

    slider.on_changed(update)

    plt.show()

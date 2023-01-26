import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchaudio
from matplotlib.widgets import SpanSelector

import disco_sound.cfg as cfg

np.random.seed(0)

from disco_sound.util import inference_utils as infer

log = logging.getLogger(__name__)


def add_example(
    label_list,
    wav_file,
    begin_idx,
    end_idx,
    sound_type,
    hop_length=None,
    sample_rate=None,
):
    """
    Adds an example to the label list.
    :param label_list:
    :param wav_file:
    :param begin_idx:
    :param end_idx:
    :param sound_type:
    :param hop_length:
    :param sample_rate:
    :return:
    """
    begin_time = infer.convert_spectrogram_index_to_seconds(
        begin_idx, hop_length=hop_length, sample_rate=sample_rate
    )
    end_time = infer.convert_spectrogram_index_to_seconds(
        end_idx, hop_length=hop_length, sample_rate=sample_rate
    )
    label_list.append(
        {
            "Begin Time (s)": begin_time,
            "End Time (s)": end_time,
            "Sound_Type": sound_type.upper(),
            "Filename": wav_file,
        }
    )


class SimpleLabeler:
    """
    This class uses matplotlib widgets to label a spectrogram.
    """

    def __init__(
        self, wav_file, output_csv_path, key_to_label, visualization_n_fft, vertical_cut
    ):

        self.wav_file = wav_file
        self.output_csv_path = output_csv_path
        self.forbidden_keys = ("a", "t", "g", "j", "d", "c", "v", "q")
        self.key_to_label = key_to_label

        for key in set(key_to_label.keys()):
            if key in self.forbidden_keys:
                raise ValueError(
                    f"Cannot override default keypress {key}, change"
                    " value in config file"
                )

        if not os.path.isdir(os.path.dirname(os.path.abspath(self.output_csv_path))):
            os.makedirs(os.path.dirname(os.path.abspath(self.output_csv_path)))

        n = 0
        self.label_list = []

        if os.path.isfile(self.output_csv_path):
            log.info(
                f"Already labeled this wav file with label file {self.output_csv_path}. Loading labels."
            )
            df = pd.read_csv(self.output_csv_path)
            for _, row in df.iterrows():
                self.label_list.append(row.to_dict())

        self.waveform, self.sample_rate = infer.load_wav_file(self.wav_file)

        self.hop_length = 200

        self.spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=visualization_n_fft,
            hop_length=self.hop_length,
            n_mels=110,
        )(self.waveform).squeeze()

        self.spectrogram[self.spectrogram == 0] = 1
        self.vertical_cut = vertical_cut
        if self.spectrogram.shape[0] != 1 and self.spectrogram.ndim > 2:
            self.spectrogram = self.spectrogram[0, self.vertical_cut :].log2()
        else:
            self.spectrogram = self.spectrogram[self.vertical_cut :].log2()

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, figsize=(8, 6))

        self.fig.canvas.manager.set_window_title("DISCO labeling app")
        self.ax2.set_title(self.wav_file)

        self.n = n
        self.xmin = 0
        self.xmax = 0
        self.interval = 400
        self.ax1.imshow(
            self.spectrogram[self.vertical_cut :, self.n : self.n + self.interval],
            origin="upper",
        )
        self.ax1.set_yticks([])

        self.ax1.axis("off")
        self.ax2.axis("off")

        self.ax1.set_title(
            "Press left mouse button and drag "
            "to select a region in the top graph. "
            # "{:d} percent through spectrogram".format(
            #     int(self.n / self.spectrogram.shape[-1])
        )

        # plt.subplots_adjust(left=0.25)

        self.fig.canvas.mpl_connect("key_press_event", self.process_keystroke)

        self.span = SpanSelector(
            self.ax1,
            self.onselect,
            "horizontal",
            useblit=True,
            rectprops=dict(alpha=0.5, facecolor="tab:blue"),
        )
        self._redraw_ax1()

    def show(self):
        self.fig.canvas.draw()
        plt.show()

    def save_labels(self):
        if len(self.label_list):
            label_df = pd.DataFrame.from_dict(self.label_list)
            label_df.to_csv(
                self.output_csv_path,
                index=False,
                mode="a" if os.path.isfile(self.output_csv_path) else "w",
                header=False if os.path.isfile(self.output_csv_path) else True,
            )

    def _redraw_ax2(self):
        self.ax2.imshow(
            self.spectrogram[
                self.vertical_cut :, self.n + self.xmin : self.n + self.xmax
            ],
            origin="upper",
        )
        self.ax2.set_title("selected region")
        self.fig.canvas.draw()

    def _redraw_ax1(self):
        self.ax1.clear()
        self.ax1.imshow(
            self.spectrogram[self.vertical_cut :, self.n : self.n + self.interval],
            aspect="auto",
            origin="upper",
        )
        self.ax1.set_title(
            "Press left mouse button and drag "
            "to select a region in the top graph "
            #             "{:d} percent through spectrogram".format(
            #                 int(100 * self.n / self.spectrogram.shape[-1])
            #             )
        )
        if len(self.label_list) != 0:
            for label in self.label_list:
                cls = label["Sound_Type"]
                begin = infer.convert_time_to_spect_index(
                    label["Begin Time (s)"],
                    hop_length=self.hop_length,
                    sample_rate=self.sample_rate,
                )
                end = infer.convert_time_to_spect_index(
                    label["End Time (s)"],
                    hop_length=self.hop_length,
                    sample_rate=self.sample_rate,
                )

                # if a label intersects the current viewing window, plot it up to the end of the window.
                # if I start after the start of the current window
                # and if my end is near the current end i
                if begin >= (self.n - 200) and end <= (self.n + self.interval + 200):
                    # then plot me
                    # starting from begin;
                    start_plot = max(begin - self.n, 0)
                    # ending either at interval or the true end (because coordinates are relative)
                    end_plot = min(end - self.n, self.interval)
                    self.ax1.plot(
                        range(start_plot, end_plot),
                        (end_plot - start_plot) * [1],
                        color=cfg.name_to_rgb_code[cls],
                        linewidth=12,
                    )

        self.fig.canvas.draw()

    def onselect(self, x_min, x_max):
        self.xmin = int(x_min)
        self.xmax = int(x_max)
        if (self.xmax - self.xmin) >= 2:
            self._redraw_ax2()

    def process_keystroke(self, key):

        if key.key in set(self.key_to_label.keys()):
            log.info(f"saving {self.key_to_label[key.key]} chirp (r to delete)")
            add_example(
                self.label_list,
                self.wav_file,
                self.n + self.xmin,
                self.n + self.xmax,
                sound_type=self.key_to_label[key.key],
                hop_length=self.hop_length,
                sample_rate=self.sample_rate,
            )
        elif key.key == "r":
            if len(self.label_list):
                self.label_list.pop()
                log.info("deleting last selection")
            else:
                log.info(
                    "empty label list! hit <A, B, X> after selecting a region"
                    " to add a labeled region to the list"
                )
        elif key.key == "a":
            log.info("widening window")
            self.interval += 10
            self._redraw_ax1()
        elif key.key == "t":
            log.info("tightening window")
            self.interval -= 10
            self._redraw_ax1()
        elif key.key == "g":
            self.n = self.n + self.interval // 2
            self._redraw_ax1()
        elif key.key == "j":
            self.n = int(self.spectrogram.shape[-1] * np.random.rand())
            self._redraw_ax1()
        elif key.key == "d":
            log.info("reversing window")
            self.n = self.n - self.interval // 2
            self._redraw_ax1()
        elif key.key == "c":
            log.info("shifting top limit down")
            self.vertical_cut = self.vertical_cut - 1
            self._redraw_ax1()
        elif key.key == "v":
            log.info("shifting top limit up")
            self.vertical_cut = self.vertical_cut + 1
            self._redraw_ax1()
        elif key.key == "q":
            log.info("saving and quitting")
        else:
            log.info("unknown key pressed")

import matplotlib
import matplotlib.pyplot as plt
import torchaudio
import numpy as np
import pandas as pd
import os
from argparse import ArgumentParser
from matplotlib.widgets import SpanSelector

np.random.seed(19680801)

import beetles.inference_utils as infer
from beetles.config import Config


def add_example(
    label_list,
    wav_file,
    begin_idx,
    end_idx,
    sound_type,
    hop_length=None,
    sample_rate=None,
):
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
    def __init__(self, wav_file, output_csv_path, config):

        self.wav_file = wav_file
        self.output_csv_path = output_csv_path
        self.config = config
        self.forbidden_keys = ("a", "t", "g", "j", "d", "c", "v", "q")

        for key in self.config.label_keys:
            if key in self.forbidden_keys:
                raise ValueError(
                    f"Cannot override default keypress {key}, change"
                    " value in config file"
                )

        if not os.path.isdir(os.path.dirname(os.path.abspath(self.output_csv_path))):
            os.makedirs(os.path.dirname(os.path.abspath(self.output_csv_path)))

        n = 0

        if os.path.isfile(self.output_csv_path):
            try:
                previous_labels = pd.read_csv(output_csv_path, delimiter=",")
                n = int(previous_labels["End Time (s)"].iloc[-1])
                print(
                    "Already labeled this wav file, with {} labels!".format(
                        previous_labels.shape[0]
                    )
                )
                print("setting initial window to last labeled chirp!")
            except:
                n = 0
                print(
                    "labels exist but couldn't parse columns, starting from beginning"
                )

        self.waveform, self.sample_rate = infer.load_wav_file(self.wav_file)

        self.hop_length = 200

        self.spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=config.visualization_n_fft,
            hop_length=self.hop_length,
        )(self.waveform).squeeze()

        self.spectrogram[self.spectrogram == 0] = 1
        self.vertical_cut = config.vertical_cut
        self.spectrogram = self.spectrogram[self.vertical_cut :].log2()

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, figsize=(8, 6))

        self.n = n
        self.xmin = 0
        self.xmax = 0
        self.interval = 400
        self.label_list = []

        self.ax1.imshow(
            self.spectrogram[self.vertical_cut :, self.n : self.n + self.interval],
            origin="upper",
        )
        self.ax1.axis("off")
        self.ax2.axis("off")

        self.ax1.set_title(
            "Press left mouse button and drag "
            "to select a region in the top graph. "
            "{:d} percent through spectrogram".format(
                int(self.n / self.spectrogram.shape[-1])
            )
        )

        textstr = (
            "keys control which label is\n"
            "assigned to the selected region.\n"
            "first navigate with <g,d,j> over\n"
            "the spectrogram, then click and\n"
            "drag to select a region.\n"
            "The selected region will appear\n"
            "on the bottom plot. If it looks good,\n"
            "save it by pressing the keys "
            f"{self.config.label_keys} (defined in config.py,"
            f" or in your custom config.yaml)>.\n"
            "Closing the window will save the labels.\n "
            "key:\n"
            "r: delete last label\n"
            "a: widen window\n"
            "t: tighten window\n"
            "g: move window right\n"
            "d: move window left\n"
            "j: jump 10 windows forward\n\n"
        )

        plt.figtext(0.02, 0.25, textstr, fontsize=8)
        plt.subplots_adjust(left=0.25)

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
                header=False,
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
        # could be
        self.ax1.clear()
        self.ax1.imshow(
            self.spectrogram[self.vertical_cut :, self.n : self.n + self.interval],
            aspect="auto",
            origin="upper",
        )
        self.ax1.set_title(
            "Press left mouse button and drag "
            "to select a region in the top graph "
            "{:d} percent through spectrogram".format(
                int(100 * self.n / self.spectrogram.shape[-1])
            )
        )
        self.fig.canvas.draw()

    def onselect(self, x_min, x_max):
        self.xmin = int(x_min)
        self.xmax = int(x_max)
        if (self.xmax - self.xmin) >= 2:
            self._redraw_ax2()

    def process_keystroke(self, key):

        if key.key in self.config.label_keys:
            print(f"saving {self.config.key_to_label[key.key]} chirp (r to delete)")
            add_example(
                self.label_list,
                self.wav_file,
                self.n + self.xmin,
                self.n + self.xmax,
                sound_type=self.config.key_to_label[key.key],
                hop_length=self.hop_length,
                sample_rate=self.sample_rate,
            )
        elif key.key == "r":
            if len(self.label_list):
                self.label_list.pop()
                print("deleting last selection")
            else:
                print(
                    "empty label list! hit <A, B, X> after selecting a region"
                    " to add a labeled region to the list"
                )
        elif key.key == "a":
            print("widening window")
            self.interval += 10
            self._redraw_ax1()
        elif key.key == "t":
            print("tightening window")
            self.interval -= 10
            self._redraw_ax1()
        elif key.key == "g":
            self.n = self.n + self.interval // 2
            self._redraw_ax1()
        elif key.key == "j":
            self.n = int(self.spectrogram.shape[-1] * np.random.rand())
            self._redraw_ax1()
        elif key.key == "d":
            print("reversing window")
            self.n = self.n - self.interval // 2
            self._redraw_ax1()
        elif key.key == "c":
            print("shifting top limit down")
            self.vertical_cut = self.vertical_cut - 1
            self._redraw_ax1()
        elif key.key == "v":
            print("shifting top limit up")
            self.vertical_cut = self.vertical_cut + 1
            self._redraw_ax1()
        elif key.key == "q":
            print("saving and quitting")
        else:
            print("unknown key pressed")


def label(config, wav_file, output_csv_path):

    labeler = SimpleLabeler(wav_file, output_csv_path, config=config)
    plt.show()
    labeler.show()
    labeler.save_labels()

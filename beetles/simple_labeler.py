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
    def __init__(self, wav_file, output_csv_path):

        self.wav_file = wav_file
        self.output_csv_path = output_csv_path
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
            n_fft=1250,
            f_min=1000,
            hop_length=self.hop_length,
        )(self.waveform)
        self.spectrogram[self.spectrogram == 0] = 1
        self.spectrogram = self.spectrogram.log2().numpy().squeeze()
        self.vertical_cut = 0

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, figsize=(8, 6))

        self.n = n
        self.xmin = 0
        self.xmax = 0
        self.interval = 400
        self.label_list = []

        self.ax1.imshow(
            self.spectrogram[self.vertical_cut :, self.n : self.n + self.interval]
        )

        self.ax1.set_title(
            "Press left mouse button and drag "
            "to select a region in the top graph "
            "{:d} percent through spectrogram".format(
                int(self.n / self.spectrogram.shape[-1])
            )
        )

        textstr = (
            "keys control which label is\n"
            "assigned to the selected region.\n"
            "first navigate with <f,d,j> over\n"
            "the spectrogram, then click and\n"
            "drag to select a region.\n"
            "The selected region will appear\n"
            "on the bottom plot. If it looks good,\n"
            "save it with <y,w,e>.\n"
            "Closing the window will save the labels.\n "
            "key:\n"
            "y: save A chirp\n"
            "w: save B chirp\n"
            "e: save background\n"
            "r: delete last label\n"
            "a: widen window\n"
            "t: tighten window\n"
            "f: move window right\n"
            "d: move window left\n"
            "j: jump 10 windows forward\n\n"
            "disclaimer: this is not production\n"
            "code and has severe limitations\n"
            "but should work in certain scenarios."
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
            if os.path.isfile(self.output_csv_path):
                label_df.to_csv(
                    self.output_csv_path, index=False, mode="a", header=False
                )
            else:
                label_df.to_csv(self.output_csv_path, index=False, mode="w")

    def _redraw_ax2(self):
        self.ax2.imshow(
            self.spectrogram[
                self.vertical_cut :, self.n + self.xmin : self.n + self.xmax
            ]
        )
        self.ax2.set_title("selected region")
        self.fig.canvas.draw()

    def _redraw_ax1(self):
        # could be
        self.ax1.clear()
        self.ax1.imshow(
            self.spectrogram[self.vertical_cut :, self.n : self.n + self.interval],
            aspect="auto",
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

        if key.key in ("y", "Y"):
            print("saving A chirp (r to delete)")
            add_example(
                self.label_list,
                self.wav_file,
                self.n + self.xmin,
                self.n + self.xmax,
                "A",
                hop_length=self.hop_length,
                sample_rate=self.sample_rate,
            )
        elif key.key in ("w", "W"):
            print("saving B chirp (r to delete)")
            add_example(
                self.label_list,
                self.wav_file,
                self.n + self.xmin,
                self.n + self.xmax,
                "B",
                hop_length=self.hop_length,
                sample_rate=self.sample_rate,
            )
        elif key.key in ("e", "E"):
            add_example(
                self.label_list,
                self.wav_file,
                self.n + self.xmin,
                self.n + self.xmax,
                "background",
                hop_length=self.hop_length,
                sample_rate=self.sample_rate,
            )
            print("saving background chirp (r to delete)")
        elif key.key in ("r", "R"):
            if len(self.label_list):
                self.label_list.pop()
                print("deleting last selection")
            else:
                print(
                    "empty label list! hit <A, B, X> after selecting a region"
                    " to add a labeled region to the list"
                )
        elif key.key in ("a", "A"):
            print("widening window")
            self.interval += 10
            self._redraw_ax1()
        elif key.key in ("t", "T"):
            print("tightening window")
            self.interval -= 10
            self._redraw_ax1()
        elif key.key in ("g", "G"):
            self.n = self.n + self.interval // 2
            self._redraw_ax1()
        elif key.key in ("j", "J"):
            self.n = int(self.spectrogram.shape[-1] * np.random.rand())
            self._redraw_ax1()
        elif key.key in ("d", "D"):
            print("reversing window")
            self.n = self.n - self.interval // 2
            self._redraw_ax1()
        elif key.key in ("c", "C"):
            print("shifting top limit down")
            self.vertical_cut = self.vertical_cut - 1
            self._redraw_ax1()
        elif key.key in ("v", "V"):
            print("shifting top limit up")
            self.vertical_cut = self.vertical_cut + 1
            self._redraw_ax1()
        elif key.key == "q":
            print("saving and quitting")
        else:
            print("unknown key pressed")


def main(args):
    labeler = SimpleLabeler(args.wav_file, args.output_csv_path)
    plt.show()
    labeler.show()
    labeler.save_labels()

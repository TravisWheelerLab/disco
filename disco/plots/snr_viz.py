import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")


root = "/xdisk/twheeler/colligan/ground_truth/snr_ablation"

snrs = [
    "20_ensemble_2_bootstrap",
    "30_ensemble_2_bootstrap",
    "40_ensemble_2_bootstrap",
    "0_ensemble_2_bootstrap",
]

test_directory = "180101_0133S12-viz"

ex_2 = [154242 - 100, 154783 + 100]

fig, ax = plt.subplots(
    nrows=4, figsize=(12.2, 8.2), gridspec_kw={"wspace": 0, "hspace": 0}
)
spects = []

for i, ensemble_directory in enumerate(snrs):
    ensemble_directory = os.path.join(root, ensemble_directory)
    spectrogram_path = os.path.join(
        ensemble_directory, test_directory, "raw_spectrogram.pkl"
    )
    with open(spectrogram_path, "rb") as src:
        spectrogram = pickle.load(src)

    spects.append(spectrogram)

vmin = np.min(spects[-1])
vmax = np.max(spects[-1])

noise_levels = [20, 30, 40, "no noise"]

for i, spectrogram in enumerate(spects):
    ax[i].imshow(spectrogram[:, ex_2[0] : ex_2[1]], vmin=vmin, vmax=vmax, aspect="auto")
    ax[i].set_ylabel(noise_levels[i], fontsize=16)
    # ax[i].imshow(spectrogram[:, ex_2[0]:ex_2[1]])

ax[0].set_title("spectrograms at different sn ratios", fontsize=18)

subplot = fig.add_subplot(111, frameon=False)

plt.tick_params(
    labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
)
plt.ylabel("signal-to-noise ratio", color="black", fontsize=18)
subplot.spines["top"].set_visible(False)
subplot.spines["right"].set_visible(False)
subplot.spines["bottom"].set_visible(False)
subplot.spines["left"].set_visible(False)
subplot.set_facecolor("none")
subplot.grid(alpha=0.0, color="#808080")

for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.spines["bottom"].set_color("black")
    a.spines["left"].set_color("#808080")
    a.set_facecolor("none")
    a.grid(alpha=0.0, color="#808080")
    a.set_xticks([])
    a.set_yticks([])

plt.subplots_adjust(wspace=0.00, hspace=0.00)

plt.savefig(
    f"{os.path.join(os.environ['HOME'], 'snr_visualization.pdf')}",
    format="pdf",
    dpi=600,
    bbox_inches="tight",
)

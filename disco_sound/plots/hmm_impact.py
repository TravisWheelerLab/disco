import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

plt.style.use("ggplot")
import os
import pickle

import numpy as np

import disco_sound.cfg as cfg
import disco_sound.plots.figure_utils as fig_utils


def colorfy(label_array, color_dict):
    out = np.zeros((1, label_array.shape[0], 3))
    for j, cls in enumerate(label_array):
        out[:, j, :] = color_dict[cls]

    return out


ex_1 = [193248, 193393]
ex_2 = [287578, 287792]
ex_3 = [239261, 239332]

root = os.path.join(fig_utils.root, "hmm_impact", "180101_0183S34D06-viz")
spectrogram = f"{root}/raw_spectrogram.pkl"
medians = f"{root}/median_predictions.pkl"
smoothed = f"{root}/hmm_predictions.pkl"
ground_truth_csv = f"{fig_utils.root}/{fig_utils.label_files[1]}"

with open(spectrogram, "rb") as src:
    spect = pickle.load(src)

with open(medians, "rb") as src:
    medians = pickle.load(src)

with open(smoothed, "rb") as src:
    smoothed = pickle.load(src)

longest = ex_2[1] - ex_2[0]  # this is going to be 1.
a = ex_1[1] - ex_1[0]
b = ex_3[1] - ex_3[0]

ground_truth = fig_utils.create_label_array(ground_truth_csv, spect)

fig, ax = plt.subplots(
    nrows=4,
    ncols=3,
    figsize=(12, 5),
    gridspec_kw={
        "height_ratios": [5, 1, 1, 1],
        "width_ratios": [a / longest, 1, b / longest],
    },
)

color_dict = {}
for class_code in range(len(cfg.class_code_to_name.keys())):
    class_hex_code = cfg.name_to_rgb_code[cfg.class_code_to_name[class_code]]
    class_rgb_code = np.array(to_rgb(class_hex_code))
    color_dict[class_code] = class_rgb_code

for i, ex in enumerate([ex_1, ex_2, ex_3]):
    time = ((ex[1] - ex[0]) * 200) / 48000

    ax[0, i].imshow(spect[:, ex[0] : ex[1]], aspect="auto")
    ax[0, i].set_title(f"length: {time:.2f}s")

    medians_colored = colorfy(medians[ex[0] : ex[1]], color_dict)
    smoothed_colored = colorfy(smoothed[ex[0] : ex[1]], color_dict)
    gt_colored = colorfy(ground_truth[ex[0] : ex[1]], color_dict)

    ax[1, i].imshow(medians_colored, aspect="auto", interpolation="nearest")
    ax[2, i].imshow(smoothed_colored, aspect="auto", interpolation="nearest")
    ax[3, i].imshow(gt_colored, aspect="auto", interpolation="nearest")

for a in ax:
    for y in a:
        y.set_xticks([])
        y.set_yticks([])
        y.axis("tight")

a_label = mpatches.Patch(color=color_dict[cfg.name_to_class_code["A"]], label="A")
b_label = mpatches.Patch(color=color_dict[cfg.name_to_class_code["B"]], label="B")
background_label = mpatches.Patch(
    color=color_dict[cfg.name_to_class_code["BACKGROUND"]], label="background"
)

legend = ax[0, 0].legend(
    loc="lower left",
    fancybox=True,
    labelcolor="black",
    framealpha=0.0,
    fontsize=16,
    handles=[a_label, b_label, background_label],
)

fig.text(y=0.33, x=0.06, s="ensemble\nprediction")
fig.text(y=0.24, x=0.06, s="post HMM")
fig.text(y=0.15, x=0.07, s="human")

ax[2, 0].set_xlabel("B chirp", fontsize=16)
ax[2, 1].set_xlabel("B chirp", fontsize=16)
ax[2, 2].set_xlabel("A chirp", fontsize=16)

fig.align_labels()
plt.subplots_adjust(wspace=0.00, hspace=0.0)

plt.savefig(
    f"{os.path.join(os.environ['HOME'], 'hmm_impact.pdf')}",
    format="pdf",
    bbox_inches="tight",
    dpi=600,
)

plt.show()

import os
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

# plt.style.use("ggplot")
import disco_sound.cfg as cfg
import disco_sound.plots.figure_utils as fig_utils


def colorfy(label_array, color_dict):
    out = np.zeros((1, label_array.shape[0], 3))
    for j, cls in enumerate(label_array):
        out[:, j, :] = color_dict[cls]

    return out


color_dict = {}
for class_code in range(len(cfg.class_code_to_name.keys())):
    class_hex_code = cfg.name_to_rgb_code[cfg.class_code_to_name[class_code]]
    class_rgb_code = np.array(to_rgb(class_hex_code))
    color_dict[class_code] = class_rgb_code

# qa'd with listening
ex_1 = [112034, 112672]
# good example with two chirps that I did _not_ call As
# ex_2 = [112534, 113665] -> original
# ex_2 = [112534-10000, 113665]
ex_2 = [154242 - 100, 154783 + 100]

ex_1 = [18600 - 100, 18700 + 100]
# ex_2 = [2723, 3164]
# ex_2 = [1723, 4164]
second_start = (ex_2[0] * 200) / 48000
second_end = (ex_2[1] * 200) / 48000
first_start = (ex_1[0] * 200) / 48000
first_end = (ex_1[1] * 200) / 48000
print(second_start, second_end)
print(first_start, first_end)

root = os.path.join(
    fig_utils.root,
    "compare_init_techniques",
    "more_ensembles",
    "random_init",
    "UNet1D",
    "ensemble_1",
    "180101_0133S12-viz/",
)

spectrogram = f"{root}/raw_spectrogram.pkl"
medians = f"{root}/median_predictions.pkl"
smoothed = f"{root}/hmm_predictions.pkl"
ground_truth_csv = f"{fig_utils.root}/{fig_utils.label_files[0]}"

with open(spectrogram, "rb") as src:
    spect = pickle.load(src)

with open(medians, "rb") as src:
    medians = pickle.load(src)

with open(smoothed, "rb") as src:
    smoothed = pickle.load(src)

ground_truth = fig_utils.create_label_array(ground_truth_csv, spect)

boundaries = np.where(np.diff(medians) != 0)[0] + 1
# for i in range(len(boundaries) - 1):
#     fig, ax = plt.subplots(nrows=3)
#     ax[0].imshow(spect[:, boundaries[i]-100:boundaries[i+1]+100], aspect="auto")
#     medians_colored = colorfy(medians[boundaries[i]-100:boundaries[i+1]+100], color_dict=color_dict)
#     ground_truth_colored = colorfy(ground_truth[boundaries[i]-100:boundaries[i+1]+100], color_dict=color_dict)
#     ax[1].imshow(medians_colored, aspect="auto")
#     ax[2].imshow(ground_truth_colored, aspect="auto")
#     s1 = (boundaries[i]-100)*200 / 48000
#     s2 = (boundaries[i+1]+100)*200 / 48000
#     ax[0].set_title(f"{boundaries[i]-100}:{boundaries[i+1]+100}, {s1//60:.3f}:{s1%60:.3f}, {s2//60:.3f}:{s2%60:.3f}")
#     plt.show()

longest = ex_2[1] - ex_2[0]  # this is going to be 1.
a = ex_1[1] - ex_1[0]


fig, ax = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(12, 5),
    gridspec_kw={"height_ratios": [5, 1, 1], "width_ratios": [a / longest, 1]},
)


for i, ex in enumerate([ex_1, ex_2]):
    time = ((ex[1] - ex[0]) * 200) / 48000

    ax[0, i].imshow(spect[:, ex[0] : ex[1]], aspect="auto")
    ax[0, i].set_title(f"length: {time:.2f}s")

    medians_colored = colorfy(medians[ex[0] : ex[1]], color_dict)
    smoothed_colored = colorfy(smoothed[ex[0] : ex[1]], color_dict)
    gt_colored = colorfy(ground_truth[ex[0] : ex[1]], color_dict)

    ax[1, i].imshow(medians_colored, aspect="auto", interpolation="nearest")
    # ax[2, i].imshow(smoothed_colored,
    #                 aspect="auto", interpolation="nearest")
    ax[2, i].imshow(gt_colored, aspect="auto", interpolation="nearest")

for a in ax:
    for y in a:
        y.set_xticks([])
        y.set_yticks([])
        y.axis("tight")

a_label = mpatches.Patch(color=color_dict[cfg.name_to_class_code["A"]], label="A")
# b_label = mpatches.Patch(color=color_dict[cfg.name_to_class_code["B"]], label="B")
background_label = mpatches.Patch(
    color=color_dict[cfg.name_to_class_code["BACKGROUND"]], label="background"
)

legend = ax[0, 0].legend(
    loc=(0.0, 0.02),
    fancybox=True,
    labelcolor="black",
    framealpha=0.0,
    fontsize=16,
    handles=[a_label, background_label],
)

fig.text(y=0.25, x=0.08, s="DISCO")
fig.text(y=0.15, x=0.08, s="human")

# ax[2, 2].set_xlabel("A chirp", fontsize=16)

fig.align_labels()
plt.subplots_adjust(wspace=0.00, hspace=0.0)

plt.savefig(
    f"{os.path.join(os.environ['HOME'], 'fp_is_tp.pdf')}",
    format="pdf",
    bbox_inches="tight",
    dpi=600,
)

plt.show()

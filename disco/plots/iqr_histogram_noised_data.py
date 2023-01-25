import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")
import numpy as np

from disco import root

root = os.path.join(
    root, "resources/noised_iqr/snr_{}_ensemble_{}_random_init/iqrs.pkl"
)

# snrs = [15, 20, 25, 30, 35, 40, 80, 160, 320, 0]
snrs = [15, 20, 25, 30, 35, 40, 80, 0]


data = []

for ensemble in [10]:
    for i, snr in enumerate(snrs):
        with open(root.format(snr, ensemble), "rb") as src:
            iqr = pickle.load(src)

        data.append(np.max(iqr, axis=0))

snrs[-1] = "no\nnoise"
fig, ax = plt.subplots(nrows=len(data), sharey=True, sharex=True, figsize=(12.2, 8.2))

for s, d, a in zip(snrs, data, ax):
    sns.violinplot(x=d, ax=a, cut=0, linewidth=1, color="lightblue")
    a.set_xlim(-0.01, 1.0)
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.spines["bottom"].set_color("#808080")
    a.spines["left"].set_color("#808080")
    a.set_facecolor("none")
    a.set_ylabel(s)
    a.collections[0].set_edgecolor(None)

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

ax[-1].set_xlabel("max iqr", fontsize=18, color="black")

# ax[0].set_title("iqr violin plots, different SN ratios", fontsize=18)
plt.subplots_adjust(bottom=0.10)
plt.savefig(
    f"{os.path.join(os.environ['HOME'], 'noisy_iqr_violin.pdf')}",
    format="pdf",
    dpi=600,
    bbox_inches="tight",
)

plt.show()

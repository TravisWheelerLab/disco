import os
import pickle

import joypy
import numpy as np

root = "disco/resources/noised_iqr/snr_{}_ensemble_{}_random_init/iqrs.pkl"

snrs = [15, 20, 25, 30, 35, 40, 80, 160, 320, 0]


data = []

for ensemble in [10]:
    for i, snr in enumerate(snrs):
        with open(root.format(snr, ensemble), "rb") as src:
            iqr = pickle.load(src)

        data.append(np.max(iqr, axis=0))

snrs[-1] = "no noise"

fig, ax = joypy.joyplot(
    data, labels=snrs, figsize=(8.2, 8.2), overlap=2, linecolor="k", linewidth=0.5
)
for a in ax:
    a.set_xlim(-0.1, 1.01)

fig.text(
    x=0.02,
    y=0.34,
    s="signal-to-noise ratio",
    rotation="vertical",
    color="black",
    fontsize=18,
)
ax[-1].set_xlabel("max iqr", fontsize=18, color="black")
# fig.text(y=0.025, x=0.52, s="max iqr",
#          color="black", fontsize=18)

plt.suptitle("iqr histograms, different SN ratios", fontsize=18)
plt.subplots_adjust(bottom=0.10)
plt.savefig(
    f"{os.path.join(os.environ['HOME'], 'noisy_iqr_histogram.pdf')}",
    format="pdf",
    dpi=600,
    bbox_inches="tight",
)

plt.show()

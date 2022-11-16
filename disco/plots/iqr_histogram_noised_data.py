import pickle

import joypy
import matplotlib
import matplotlib.pyplot as plt
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

fig, ax = joypy.joyplot(data, labels=snrs, hist=True, bins=100)

fig.text(x=0.05, y=0.3, s="signal-to-noise ratio", rotation="vertical")
fig.text(y=0.02, x=0.5, s="max iqr")
plt.suptitle("histograms of iqr for different SN ratios")

plt.show()

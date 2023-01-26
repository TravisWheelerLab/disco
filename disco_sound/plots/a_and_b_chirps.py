import os
import pickle

import matplotlib.pyplot as plt

data_path = os.path.join(
    f"{os.environ['HOME']}",
    "disco_figure_resources",
    "a_and_b_chirps",
    "raw_spectrogram.pkl",
)

with open(data_path, "rb") as src:
    spect = pickle.load(src)


a_index = 276000
ofs = 1500

b_index = 288800
fig, ax = plt.subplots(nrows=2, figsize=(12.2, 8.2))

ax[0].imshow(spect[:, a_index : a_index + ofs], aspect="auto")
ax[1].imshow(spect[:, b_index : b_index + ofs], aspect="auto")

ax[0].set_title("A chirps")
ax[1].set_title("B chirps")
length_of_each = (ofs * 200) / 48000
print(length_of_each)

for a in ax:
    a.set_xticks([])
    a.set_yticks([])
    a.axis("tight")

plt.subplots_adjust(wspace=0, hspace=0.1)

plt.savefig(
    f"{os.path.join(os.environ['HOME'], 'a_and_b_chirps.pdf')}",
    format="pdf",
    bbox_inches="tight",
    dpi=600,
)

plt.show()

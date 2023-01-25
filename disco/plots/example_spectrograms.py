import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torchaudio

from disco import root

western_meadowlark, _ = torchaudio.load(
    os.path.join(root, "resources", "western_meadowlark.wav")
)
hello = os.path.join(root, "resources", "hello.pkl")
beetles, _ = torchaudio.load(os.path.join(root, "resources", "example.wav"))

with open(hello, "rb") as src:
    hello = pickle.load(src)
    hello[hello == 0] = 1e-10
    hello = np.log2(hello)

spectrogram = torchaudio.transforms.Spectrogram()
western_meadowlark = spectrogram(western_meadowlark)[0].log2()

beetles = spectrogram(beetles)[0].log2()

window_size = 400
hop_length = 200

# therefore for every 200 timepoints
# therefore every single element of each spectrogram
# contains

subslice_hello = hello[:, 300:]
subslice_beetles = beetles[:, 1275:1400]

# 48 Khz microphone
subslice_hello_seconds = (subslice_hello.shape[1] * hop_length) / 48000
subslice_beetles_seconds = (subslice_beetles.shape[1] * 200) / 48000
meadowlark_seconds = (western_meadowlark.shape[1] * 200) / 48000

print(subslice_hello_seconds, meadowlark_seconds, subslice_beetles_seconds)

fig, ax = plt.subplots(ncols=3, tight_layout=True)
ax[0].imshow(subslice_hello)
ax[0].set_title("hello", fontsize=14)
ax[1].imshow(western_meadowlark)
ax[1].set_title("western meadowlark", fontsize=14)
ax[2].imshow(subslice_beetles)
ax[2].set_title("beetle chirp", fontsize=14)

for a in ax:
    a.set_xticks([])
    a.set_yticks([])
    a.axis("tight")


plt.savefig(
    f"{os.path.join(os.environ['HOME'], 'hello_birds_beetle.pdf')}",
    format="pdf",
    bbox_inches="tight",
    dpi=600,
)
plt.show()

import os.path
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from disco_sound.datasets.whale_data import WhaleDataset
from disco_sound.models.unet_1d import WhaleUNet

device = torch.device("cuda")

files = glob("/xdisk/twheeler/colligan/whale_data/data/train/*")
val_files = files[int(0.8 * len(files)) :]

with open("/home/u4/colligan/val_files.txt", "w") as dst:
    for file in val_files:
        dst.write(os.path.basename(file) + "\n")

exit()

ckpt_paths = glob("/xdisk/twheeler/colligan/whale_models_jan17/WhaleUNet/*ckpt")

dataset = WhaleDataset(
    val_files,
    label_csv="/xdisk/twheeler/colligan/whale_data/data/train.csv",
    n_fft=1150,
    hop_length=20,
)

models = []
for ckpt_path in ckpt_paths:
    model = WhaleUNet.load_from_checkpoint(ckpt_path, map_location=device).to(device)
    models.append(model)

print(len(models))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

correct_mode = 0
correct_argmax = 0
total = 0
all_labels = []
all_preds = []

with torch.no_grad():

    for features, labels in dataloader:
        print(f"{total/len(dataloader)}", end="\r")
        pred_array = torch.zeros((len(models), features.shape[0], features.shape[-1]))

        for k, model in enumerate(models):
            logits = model(features.to(device)).squeeze()
            preds = torch.sigmoid(logits).to("cpu")
            pred_array[k] = preds

        # grab the median sigmoid over the model dimension
        median_sigmoids = np.median(pred_array, axis=0)
        # then round the simgiod for predictions
        modes, _ = torch.mode(torch.round(torch.Tensor(median_sigmoids)))

        for kk, (pred, label) in enumerate(zip(modes.squeeze(), labels)):
            if pred == 0 and label == 0:
                true_neg = features[kk]
            elif pred == 0 and label == 1:
                false_neg = features[kk]
            elif pred == 1 and label == 1:
                true_pos = features[kk]
            elif pred == 1 and label == 0:
                false_pos = features[kk]

        correct_mode += torch.sum(modes.to("cpu").squeeze() == labels)
        all_preds.append(modes.to("cpu").squeeze())
        all_labels.append(labels)
        total += labels.shape[0]


preds = torch.cat(all_preds)
labels = torch.cat(all_labels)
acc = torch.sum(preds == labels) / labels.shape[0]

print(acc)
cmat = confusion_matrix(labels, preds)
print(cmat / np.sum(cmat, axis=0, keepdims=True))
print(cmat / np.sum(cmat, axis=1, keepdims=True))
print(cmat)

vmax = max(
    np.max(np.log2(true_pos.numpy())),
    np.max(np.log2(false_pos.numpy())),
    np.max(np.log2(false_neg.numpy())),
    np.max(np.log2(true_neg.numpy())),
)
vmin = min(
    np.min(np.log2(true_pos.numpy())),
    np.min(np.log2(false_pos.numpy())),
    np.min(np.log2(false_neg.numpy())),
    np.min(np.log2(true_neg.numpy())),
)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12.2, 8.2))
ax[0, 0].imshow(np.log2(true_pos), aspect="auto", vmin=vmin, vmax=vmax)
ax[0, 1].imshow(np.log2(false_neg), aspect="auto", vmin=vmin, vmax=vmax)
ax[1, 0].imshow(np.log2(false_pos), aspect="auto", vmin=vmin, vmax=vmax)
ax[1, 1].imshow(np.log2(true_neg), aspect="auto", vmin=vmin, vmax=vmax)
ax[0, 0].set_title("true", fontsize=22)
ax[0, 1].set_title("false", fontsize=22)

ax[0, 0].set_ylabel("true", fontsize=22)
ax[1, 0].set_ylabel("false", fontsize=22)


plt.subplots_adjust(wspace=0.00, hspace=0.0)
plt.suptitle("predicted", fontsize=22)
fig.text(x=0.05, y=0.45, s="actual", rotation="vertical", color="black", fontsize=22)

for a in ax:
    for y in a:
        y.set_xticks([])
        y.set_yticks([])
        y.axis("tight")

plt.savefig("/home/u4/colligan/whale.pdf", bbox_inches="tight", format="pdf", dpi=600)

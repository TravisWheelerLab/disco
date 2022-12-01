import pdb
from glob import glob

import sklearn
import torch

from disco.datasets.whale_data import WhaleDataset
from disco.models.unet_1d import WhaleUNet

device = torch.device("cuda")

files = glob("/xdisk/twheeler/colligan/whale_data/data/train/*")
val_files = files[int(0.8 * len(files)) :]
ckpt_path = (
    "/xdisk/twheeler/colligan/whale_models/WhaleUNet/9/checkpoints/best_loss_model.ckpt"
)
dataset = WhaleDataset(
    val_files,
    label_csv="/xdisk/twheeler/colligan/whale_data/data/train.csv",
    n_fft=1150,
    hop_length=20,
)

model = WhaleUNet.load_from_checkpoint(ckpt_path, map_location=device).to(device)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

correct_mode = 0
correct_argmax = 0
total = 0
all_labels = []
all_preds = []

with torch.no_grad():

    for features, labels in dataloader:
        try:
            print(f"{total/len(dataloader)}", end="\r")
            logits = model(features.to(device))
            preds = torch.sigmoid(logits)
            modes, _ = torch.mode(torch.round(preds))
            correct_mode += torch.sum(modes.to("cpu").squeeze() == labels)
            all_preds.append(modes.to("cpu").squeeze())
            all_labels.append(labels)
            total += labels.shape[0]

        except RuntimeError as e:
            pdb.set_trace()

preds = torch.cat(all_preds)
labels = torch.cat(all_labels)
cmat = sklearn.metrics.confusion_matrix(labels, preds)
acc = torch.sum(preds == labels) / labels.shape[0]

print(cmat)
print(acc)

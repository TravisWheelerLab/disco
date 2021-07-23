from data_feeder import SpectrogramDataset
import torch
import train_model as tm
import pdb
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib

INDEX_TO_LABEL = {0: 'A', 1: 'B', 2: 'X'}


def save_sample(i, spectrogram, predicted_classes):
    plt.imshow(spectrogram.numpy().squeeze())
    plt.scatter(np.arange(len(predicted_classes.squeeze())),
                np.asarray(predicted_classes[:]).squeeze(),
                c=predicted_classes.squeeze(),
                cmap='rainbow',
                s=3)
    plt.colorbar()
    plt.savefig('image_offload/' + 'prediction' + str(i) + '.png')
    plt.close()
    print(i, "saved.")


if __name__ == '__main__':

    spectrograms_list = []
    root = os.path.join('test_data', 'spect')
    files = glob(os.path.join(root, "*"))
    numbers_list = list(np.random.randint(low=0, high=len(files), size=2))
    i = 0

    test_loader = torch.utils.data.DataLoader(SpectrogramDataset(directory_name='test_data', clip_spects=False),
                                              batch_size=1,
                                              shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = "beetles_cnn_for_sure_working.pt"

    model = tm.FCNNSmaller().to(device)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()

    cmap = matplotlib.colors.ListedColormap(['hotpink', 'orange', 'aquamarine'])

    save_spect_image_samples = True

    conf_mt = torch.zeros(3, 3, dtype=torch.int64)

    with torch.no_grad():
        # 1. convert spectrograms for evaluation into torch tensors
        # 2. load the saved model with torch.load (oh I did that)
        # 3. get predictions with model(spectrogram)
        i = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
            if i < 15 and save_spect_image_samples:
                save_sample(i, data, pred)
            i += 1

            stacked = torch.stack((target, pred), dim=1)
            # breakpoint()

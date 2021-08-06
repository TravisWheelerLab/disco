import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
from data_feeder import SpectrogramDataset
import confusion_matrix as cm
import spectrogram_analysis as sa

from argparse import ArgumentParser


class CNN1D(nn.Module):

    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=113, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 512, 3, padding=1)
        self.conv3 = nn.Conv1d(512, 512, 3, padding=1)
        self.conv4 = nn.Conv1d(512, 1024, 3, padding=1)
        self.conv5 = nn.Conv1d(1024, 512, 1, padding=0)
        self.conv6 = nn.Conv1d(512, 512, 3, padding=1)
        self.conv7 = nn.Conv1d(512, 256, 3, padding=1)
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=32, kernel_size=1, padding=0)
        self.conv9 = nn.Conv1d(in_channels=32, out_channels=3, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.conv9(x)
        # todo: add one more conv before output. it is 1d.
        output = F.log_softmax(x, dim=1)
        return output


class FCNNSmaller(nn.Module):

    def __init__(self):
        super(FCNNSmaller, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.maxPoolColDim = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 128, 1, padding=0)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=32, kernel_size=1, padding=0)
        self.conv9 = nn.Conv1d(in_channels=32, out_channels=3, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxPoolColDim(x)
        x = self.conv4(x)
        x = torch.mean(x, dim=2)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.conv9(x)
        # todo: add one more conv before output. it is 1d.
        output = F.log_softmax(x, dim=1)
        return output


class FCNN(nn.Module):

    def __init__(self):
        super(FCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.maxPoolColDim = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 128, 1, padding=0)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=32, kernel_size=1, padding=0)
        self.conv9 = nn.Conv1d(in_channels=32, out_channels=3, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxPoolColDim(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = torch.mean(x, dim=2)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.conv9(x)
        # todo: add one more conv before output. it is 1d.
        output = F.log_softmax(x, dim=1)
        return output


class NetSinglePrediction(nn.Module):

    def __init__(self):
        super(FCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(66880, 128)  # ONLY GOING TO WORK ON IMAGES OF SIZE 1x28x28
        # we have an image size of 1x128x40. After two convolutions and 
        # and one max pooling, we'll have an image size of 64x62x18
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch, log_interval=2):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        correct += torch.sum(output.argmax(dim=1) == target)
        total += torch.numel(target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Acc: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), correct / total))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    conf_mat = cm.ConfusionMatrix()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
            correct += torch.sum(pred == target)
            total += torch.numel(target)
            conf_mat.increment(target, pred, device)
    # conf_mat.plot(classes=['A', 'B', 'X'], save_images=False)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))


def overfit_on_batch(model, device, data_loader):
    for b in data_loader:
        features, labels = b
        break
    optim = torch.optim.Adam(model.parameters())
    features, labels = features.to(device), labels.to(device)
    for _ in range(10000):
        optim.zero_grad()
        logits = model(features)
        loss = F.nll_loss(logits, labels, reduction='sum')
        loss.backward()
        optim.step()
        preds = logits.argmax(dim=1)
        correct = torch.sum(preds == labels)
        total = torch.numel(labels)
        print('overfit accuracy: {}'.format(correct/total))

def parser():

    ap = ArgumentParser()
    ap.add_argument('--batch-size', required=True, type=int)
    ap.add_argument('--shuffle', action='store_true', required=True)
    ap.add_argument('--learning_rate', type=float, required=True)
    ap.add_argument('--epochs', type=int, required=True)
    ap.add_argument('--save_model', action='store_true', required=True)
    ap.add_argument('--overfit', action='store_true')
    ap.add_argument('--mel', action='store_true', required=True)
    ap.add_argument('--log', action='store_true', required=True)
    ap.add_argument('--n_fft', type=int, required=True)
    ap.add_argument('--vert_trim', type=int, required=True)
    ap.add_argument('--bin_spects', action='store_true')

    return ap.parse_args()



def main(args):

    batch_size = args.batch_size
    shuffle = args.shuffle
    learning_rate = args.learning_rate
    epochs = args.epochs
    save_model = args.save_model
    overfit = args.overfit
    mel = args.mel
    log = args.log
    n_fft = args.n_fft
    vert_trim = args.vert_trim
    bin_spects = args.bin_spects

    if vert_trim is None:
        vert_trim = sa.determine_default_vert_trim(mel, log, n_fft)

    spect_type = sa.form_spectrogram_type(mel, n_fft, log, vert_trim)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = SpectrogramDataset(dataset_type='train',
                                       spect_type=spect_type,
                                       batch_size=batch_size,
                                       bin_spects=bin_spects)

    test_dataset = SpectrogramDataset(dataset_type='test',
                                      spect_type=spect_type,
                                      clip_spects=False,
                                      batch_size=batch_size)
    print("datasets loaded in.")

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=shuffle)
    print("dataloaders created.")

    model = CNN1D().to(device)

    if overfit:
        overfit_on_batch(model, device, train_loader)
        exit()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        # test(model, device, test_loader)

    if save_model:
        torch.save(model.state_dict(), 'beetles_cnn_1D_' + spect_type + '_' + str(epochs) + '_epochs.pt')
        print('saved model.')


if __name__ == '__main__':
    args = parser()
    main(args)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
from data_feeder import SpectrogramDataset
import confusion_matrix as cm
import spectrogram_analysis as sa
import os


class CNN1D(nn.Module):

    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=98, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 512, 3, padding=1)
        self.conv3 = nn.Conv1d(512, 512, 3, padding=1)
        self.conv4 = nn.Conv1d(512, 1024, 3, padding=1)
        self.conv5 = nn.Conv1d(1024, 512, 1, padding=0)
        self.conv6 = nn.Conv1d(512, 512, 3, padding=1)
        self.conv7 = nn.Conv1d(512, 256, 3, padding=1)
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=32, kernel_size=1, padding=0)
        self.conv9 = nn.Conv1d(in_channels=32, out_channels=3, kernel_size=1, padding=0)

        self.lowest_test_loss = 9999999999999
        self.epoch_of_lowest_test_loss = None
        self.accuracy_lowest_test_loss = None

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


def test(model, device, test_loader, save_model, spect_type, epoch, ensemble, model_number):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
            correct += torch.sum(pred == target)
            total += torch.numel(target)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))

    if save_model:
        if test_loss < model.lowest_test_loss:
            if ensemble:
                directory_savepath = os.path.join('models', spect_type + '_ensemble')
                model_filename = 'm_' + str(model_number) + '.pt'
                save_path = os.path.join(directory_savepath, model_filename)
            else:
                save_path = os.path.join('models', spect_type + '.pt')
            torch.save(model.state_dict(), save_path)
            print('Saved model.\n')
            model.lowest_test_loss = test_loss
            model.epoch_of_lowest_test_loss = epoch
            model.accuracy_lowest_test_loss = correct / total


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


def main():
    batch_size = 256
    shuffle = True
    learning_rate = 1e-4
    epochs = 500
    save_model = True
    overfit = False
    mel = True
    log = True
    n_fft = 800
    vert_trim = 30
    bin_spects = True
    ensemble = True
    num_models = 10

    if not ensemble:
        num_models = 1

    if vert_trim is None:
        vert_trim = sa.determine_default_vert_trim(mel, log, n_fft)

    spect_type = sa.form_spectrogram_type(mel, n_fft, log, vert_trim)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if ensemble:
        # create a directory to save new ensembled models
        directory_path = os.path.join('models', spect_type + '_ensemble')
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print("Created directory for ensemble.")

    for model_idx in range(1, num_models+1):
        print('Training model number {}.'.format(model_idx))
        train_dataset = SpectrogramDataset(dataset_type='train',
                                           spect_type=spect_type,
                                           batch_size=batch_size,
                                           bin_spects=bin_spects,
                                           bootstrap_sample=ensemble)

        test_dataset = SpectrogramDataset(dataset_type='test',
                                          spect_type=spect_type,
                                          clip_spects=False,
                                          batch_size=batch_size,
                                          bootstrap_sample=ensemble)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   drop_last=True)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=shuffle)
        print("Dataloaders created.")

        model = CNN1D().to(device)

        if overfit:
            overfit_on_batch(model, device, train_loader)
            exit()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader, save_model, spect_type, epoch, ensemble, model_idx)

        print('Final model saved: epoch {}, loss: {:.4f} and accuracy: {:.0f}%.'.format(
            model.epoch_of_lowest_test_loss,
            model.lowest_test_loss,
            100 * model.accuracy_lowest_test_loss))


if __name__ == '__main__':
    main()

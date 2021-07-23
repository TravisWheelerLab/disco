import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
from data_feeder import SpectrogramDataset


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


def train(model, device, train_loader, optimizer, epoch,
          log_interval=2):
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
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
            correct += torch.sum(pred == target)


    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = SpectrogramDataset(directory_name='train_data')
    test_dataset = SpectrogramDataset(directory_name='test_data')

    batch_size = 32
    shuffle = True
    learning_rate = 1e-4
    epochs = 1000
    save_model = False

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    model = FCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for batch in train_loader:
        b = batch
        break
    features = b[0].to(device)
    labels = b[1].to(device)
    for _ in range(1000):

        optimizer.zero_grad()
        output = model(features)
        preds = output.argmax(dim=1)
        acc = torch.sum(preds == labels)/torch.numel(preds)
        loss= F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        print(acc, loss)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if save_model:
        torch.save(model.state_dict(), "beetles_cnn.pt")


if __name__ == '__main__':
    save_model = True
    main()

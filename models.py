import torch
import numpy as np


class CNN1D(torch.nn.Module):

    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=108, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 512, 3, padding=1)
        self.conv3 = nn.Conv1d(512, 512, 3, padding=1)
        self.conv4 = nn.Conv1d(512, 1024, 3, padding=1)
        self.conv5 = nn.Conv1d(1024, 512, 1, padding=0)
        self.conv6 = nn.Conv1d(512, 512, 3, padding=1)
        self.conv7 = nn.Conv1d(512, 256, 3, padding=1)
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=32, kernel_size=1, padding=0)
        self.conv9 = nn.Conv1d(in_channels=32, out_channels=3, kernel_size=1, padding=0)

        self.lowest_test_loss = torch.tensor(np.inf)
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

import torch
import torch.nn as nn
import logging


logger = logging.getLogger(__name__)


def conv3x3(in_chan, out_chan):
    return nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=3, padding=1)


class ResBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv1 = nn.Sequential(
            conv3x3(in_chan, out_chan),
            nn.BatchNorm2d(num_features=out_chan),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            conv3x3(out_chan, out_chan),
            nn.BatchNorm2d(num_features=out_chan)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=1),
            nn.BatchNorm2d(num_features=out_chan),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.downsample(x)
        return self.relu(out)


class Model(nn.Module):
    def __init__(self, img_size, n_channels):
        super().__init__()
        self.img_size = img_size
        self.blocks = nn.Sequential(
            conv3x3(in_chan=3, out_chan=n_channels * 2),
            nn.BatchNorm2d(num_features=n_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            conv3x3(in_chan=n_channels * 2, out_chan=n_channels * 4),
            nn.BatchNorm2d(num_features=n_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            conv3x3(in_chan=n_channels * 4, out_chan=n_channels * 8),
            nn.BatchNorm2d(num_features=n_channels * 8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            conv3x3(in_chan=n_channels * 8, out_chan=n_channels * 16),
            nn.BatchNorm2d(num_features=n_channels * 16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(img_size // 16)**2 * n_channels * 16, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=28)
        )

    def forward(self, inputs):
        return self.head(self.blocks(inputs))

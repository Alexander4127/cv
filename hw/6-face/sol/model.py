import torch
import torch.nn as nn
import logging


logger = logging.getLogger(__name__)


def conv3x3(in_chan, out_chan):
    return nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=3, padding=1)


class StBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = nn.Sequential(
            conv3x3(in_chan, out_chan),
            nn.BatchNorm2d(num_features=out_chan),
            nn.MaxPool2d(kernel_size=2)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class Model(nn.Module):
    def __init__(self, img_size, n_channels):
        super().__init__()
        self.img_size = img_size
        self.blocks = nn.Sequential(
            StBlock(3, n_channels),
            StBlock(n_channels, n_channels * 2),
            StBlock(n_channels * 2, n_channels * 4)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(img_size // 8)**2 * n_channels * 4, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=28)
        )

    def forward(self, inputs):
        return self.head(self.blocks(inputs))

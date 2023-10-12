import torch
import torch.nn as nn
import logging


logger = logging.getLogger(__name__)

class ResBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=in_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=in_chan),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_chan)
        )
        self.downsampler = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_chan),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.downsampler(x)
        return self.relu(out)


class Model(nn.Module):
    def __init__(self, img_size, n_channels):
        super().__init__()
        assert img_size % 16 == 0
        self.upsample = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            ResBlock(in_chan=n_channels, out_chan=n_channels * 2),
            nn.MaxPool2d(kernel_size=2),
            ResBlock(in_chan=n_channels * 2, out_chan=n_channels * 4),
            nn.MaxPool2d(kernel_size=2),
            ResBlock(in_chan=n_channels * 4, out_chan=n_channels * 8),
            nn.MaxPool2d(kernel_size=2),
            ResBlock(in_chan=n_channels * 8, out_chan=n_channels * 16),
            nn.MaxPool2d(kernel_size=2)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=img_size**2 * n_channels // 16, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=28)
        )

    def forward(self, inputs):
        return self.head(self.blocks(self.upsample(inputs)))

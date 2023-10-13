import torch
import torch.nn as nn
import logging


logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, img_size, n_channels):
        super().__init__()
        assert img_size % 16 == 0
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels * 2, out_channels=n_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=img_size**2 * n_channels // 16, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=28)
        )

    def forward(self, inputs):
        out = self.conv3(self.conv2(self.conv1(inputs)))
        return self.head(out)

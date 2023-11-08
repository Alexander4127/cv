import torch
import torch.nn as nn
from torchvision.models import resnet50


class Model(nn.Module):
    def __init__(self, num_classes: int = 50, load_mobile: bool = False):
        super().__init__()
        self.mobilenet = resnet50()
        if load_mobile:
            self.mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.head = nn.Linear(in_features=1000, out_features=num_classes)

    def __call__(self, img: torch.Tensor):
        return self.head(self.mobilenet(img))

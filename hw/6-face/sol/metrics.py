import torch
import numpy as np


from sol.test import pred2coord


class RealMSE:
    def __init__(self, name):
        self.name = name

    def __call__(self, pred, size, real_ans, **kwargs):
        pred = pred.detach()
        return torch.mean((real_ans - pred2coord(pred, size)) ** 2).item()

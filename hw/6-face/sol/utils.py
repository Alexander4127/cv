import torch
import numpy as np
from PIL import Image


def pred2coord(pred: torch.Tensor, size: torch.Tensor, **kwargs):
    assert pred.shape[0] == size.shape[0] and size.shape[1] == 2

    coord = pred.float().clone() + 0.5
    w = coord.shape[1]
    for idx, idx_range in enumerate([torch.arange(0, w, 2), torch.arange(1, w, 2)]):
        coord[:, idx_range] *= size[:, idx].unsqueeze(1)

    return coord


def coord2pred(ans: torch.Tensor, size: torch.Tensor, **kwargs):
    assert ans.shape[0] == size.shape[0] and size.shape[1] == 2

    pred = ans.float().clone()
    size = size.float()
    w = pred.shape[1]
    for idx, idx_range in enumerate([torch.arange(0, w, 2), torch.arange(1, w, 2)]):
        pred[:, idx_range] /= size[:, idx].unsqueeze(1)

    return pred - 0.5

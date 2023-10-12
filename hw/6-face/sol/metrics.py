import torch
import numpy as np


class RealMSE:
    def __init__(self, name):
        self.name = name

    def __call__(self, pred, size, real_ans, **kwargs):
        pred = pred.detach().cpu()
        for idx_range in [np.arange(len(pred)) % 2 == 0, np.arange(len(pred)) % 2 == 1]:
            pred[idx_range] = np.round(pred[idx_range] * size[:, 0][idx_range].reshape(-1, 1))
        return torch.mean((real_ans.detach().cpu() - pred) ** 2).item()

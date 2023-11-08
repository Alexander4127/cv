import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


class Accuracy:
    def __init__(self, name):
        self.name = name

    def __call__(self, labels, ans, **kwargs):
        return accuracy_score(ans.cpu().numpy(), labels.cpu().numpy())


class Precision:
    def __init__(self, name):
        self.name = name

    def __call__(self, labels, ans, **kwargs):
        return precision_score(ans.cpu().numpy(), labels.cpu().numpy(), average='macro')


class Recall:
    def __init__(self, name):
        self.name = name

    def __call__(self, labels, ans, **kwargs):
        return recall_score(ans.cpu().numpy(), labels.cpu().numpy(), average='macro')

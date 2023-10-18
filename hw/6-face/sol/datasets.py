import os
import pathlib

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as T


class ImageDataset(Dataset):
    def __init__(self, path, img_size, is_train, train_size=0.8):
        self._data_dir = pathlib.Path(path)
        self._resize = T.Resize([img_size, img_size])
        self._length = len([name for name in os.listdir(self._data_dir / 'images')])
        split_idx = train_test_split(np.arange(self._length), train_size=train_size)
        self._idx = split_idx[0] if is_train else split_idx[1]
        self._length = len(self._idx)
        self._train = is_train
        self._df = pd.read_csv(self._data_dir / 'gt.csv', index_col='filename')

    def __len__(self):
        return self._length

    @staticmethod
    def _read_img(path):
        img = plt.imread(path)
        if len(img.shape) == 2:
            return np.array([img, img, img])
        return np.transpose(img, axes=(2, 0, 1))

    def __getitem__(self, out_idx):
        idx = self._idx[out_idx]
        padded_idx = '0' * (5 - len(str(idx))) + str(idx) + '.jpg'
        img_path = self._data_dir / 'images' / padded_idx
        img = torch.tensor(self._read_img(img_path))

        assert img.shape[0] == 3
        real_size = img.shape[1:]
        img = self._resize(img)
        real_ans = np.array(self._df.loc[padded_idx])
        ans = real_ans.copy().astype(float)
        assert ans.shape == (28,)
        ans[np.arange(len(ans)) % 2 == 0] /= real_size[0]
        ans[np.arange(len(ans)) % 2 == 1] /= real_size[1]
        return {"img": img.float(), "ans": torch.tensor(ans).float(), "size": real_size, "real_ans": real_ans}

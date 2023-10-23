import logging
import os
import pathlib
from enum import Enum

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as T


from sol.utils import coord2pred
from sol.augments import BaseAugmentation, ColorJitter, HorizontalFlip, \
    VerticalFlip, RandomApply, SequentialAugmentation


class Mode(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, img_size: int, type_set: Mode, gt=None, train_size=0.9):
        self._image_dir = pathlib.Path(image_dir)
        self._transform = T.Compose([
            T.PILToTensor(),
            lambda t: t.float() / 255,
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._resize = T.Resize([img_size, img_size])

        if type_set == Mode.TRAIN:
            self._size_augments = SequentialAugmentation([
                # RandomApply(HorizontalFlip()),
                # RandomApply(VerticalFlip())
            ])
            self._st_augments = SequentialAugmentation([
                RandomApply(ColorJitter())
            ])

        self._idx: list = [name for name in os.listdir(image_dir) if name.endswith('.jpg')]
        self._type_set = type_set
        if type_set != Mode.TEST:
            split_idx = train_test_split(self._idx, train_size=train_size)
            self._idx = split_idx[0] if type_set == Mode.TRAIN else split_idx[1]
            assert gt is not None
            self._df: dict = gt

    def __len__(self):
        return len(self._idx)

    def __getitem__(self, out_idx):
        padded_idx = self._idx[out_idx]
        img_path = self._image_dir / padded_idx
        with Image.open(img_path) as image:
            img = self._transform(image.convert('RGB'))

        assert img.shape[0] == 3
        real_size = torch.tensor(img.shape[1:])
        d = {"filename": padded_idx, "img": img.float(), "size": real_size}

        if self._type_set == Mode.TEST:
            d["img"] = self._resize(d["img"])
            return d

        real_ans = torch.tensor(self._df[padded_idx])
        if self._type_set == Mode.TRAIN:
            d["img"], d["size"], real_ans = self._size_augments(d["img"], d["size"], real_ans)

        ans = coord2pred(real_ans.unsqueeze(0), real_size.unsqueeze(0)).squeeze().float()
        assert len(ans.shape) == 1
        d.update({"ans": ans})
        d["img"] = self._resize(d["img"])
        if self._type_set == Mode.TRAIN:
            d["img"], _, _ = self._st_augments(d["img"], d["size"], d["ans"])

        return d

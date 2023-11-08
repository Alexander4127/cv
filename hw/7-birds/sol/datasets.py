import os
from typing import Optional
import pathlib
from enum import Enum

import albumentations as A
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class Mode(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class ImageDataset(Dataset):
    def __init__(self,
                 image_dir: str,
                 type_set: Mode,
                 img_size: int = 256,
                 gt: Optional[dict] = None,
                 train_size: float = 0.9):
        self._image_dir = pathlib.Path(image_dir)

        if type_set == Mode.TRAIN:
            self._transform = A.Compose([
                # resize
                A.Resize(img_size, img_size),
                A.RandomCrop(224, 224),

                # flips
                A.Flip(p=0.5),
                A.Transpose(),
                A.ShiftScaleRotate(p=0.5),

                # blur
                A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),

                # brightness
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(),
                ], p=0.3),

                # normalize
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self._transform = A.Compose([
                A.Resize(img_size, img_size),
                A.CenterCrop(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self._idx = [name for name in os.listdir(image_dir) if name.endswith('.jpg')] \
            if gt is None else list(gt.keys())

        self._type_set = type_set
        if type_set != Mode.TEST:
            split_idx = train_test_split(self._idx, train_size=train_size, random_state=42)
            self._idx = split_idx[0] if type_set == Mode.TRAIN else split_idx[1]
            assert gt is not None
            self._df: dict = gt

    def __len__(self):
        return len(self._idx)

    def __getitem__(self, out_idx):
        padded_idx = self._idx[out_idx]
        img_path = self._image_dir / padded_idx
        with Image.open(img_path) as image:
            img = torch.from_numpy(
                self._transform(
                    image=np.array(image.convert('RGB'))
                )["image"]
            ).permute(2, 0, 1)

        assert img.shape[0] == 3 and len(img.shape) == 3, f'{img.shape}'
        d = {"filename": padded_idx, "img": img}

        if self._type_set == Mode.TEST:
            return d

        d.update({"ans": torch.tensor(self._df[padded_idx])})
        assert d["ans"].size() == ()

        return d

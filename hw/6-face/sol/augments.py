import random
import typing as tp

import torch
import torchvision.transforms as T


class BaseAugmentation:
    def __call__(self, img: torch.Tensor, size: torch.Tensor, ans: torch.Tensor):
        raise NotImplementedError()


class ColorJitter(BaseAugmentation):
    def __init__(self):
        self._aug = T.ColorJitter()

    def __call__(self, img: torch.Tensor, size: torch.Tensor, ans: torch.Tensor):
        return self._aug(img), size, ans


class HorizontalFlip(BaseAugmentation):
    def __init__(self):
        self._aug = T.RandomHorizontalFlip(p=1)

    def __call__(self, img: torch.Tensor, size: torch.Tensor, ans: torch.Tensor):
        ans = ans.clone()
        idx_range = torch.arange(0, len(ans)) % 2 == 0
        ans[idx_range] = size[1] - ans[idx_range] - 1
        return self._aug(img), size, ans


class VerticalFlip(BaseAugmentation):
    def __init__(self):
        self._aug = T.RandomVerticalFlip(p=1)

    def __call__(self, img: torch.Tensor, size: torch.Tensor, ans: torch.Tensor):
        ans = ans.clone()
        idx_range = torch.arange(0, len(ans)) % 2 == 1
        ans[idx_range] = size[0] - ans[idx_range] - 1
        return self._aug(img), size, ans


class RandomApply(BaseAugmentation):
    def __init__(self, aug: BaseAugmentation,  p: float = 0.5):
        self._aug = aug
        self._p = p

    def __call__(self, img: torch.Tensor, size: torch.Tensor, ans: torch.Tensor):
        if random.random() < self._p:
            return self._aug(img, size, ans)
        return img, size, ans


class SequentialAugmentation(BaseAugmentation):
    def __init__(self, aug: tp.List[BaseAugmentation]):
        self._aug = aug

    def __call__(self, img: torch.Tensor, size: torch.Tensor, ans: torch.Tensor):
        for aug in self._aug:
            img, size, ans = aug(img, size, ans)
        return img, size, ans

from datetime import datetime
from enum import Enum
import logging
import pathlib
import os
import random
import sys
import typing as tp
from typing import Dict
import warnings


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


from sol.datasets import ImageDataset, Mode
from sol.trainer import Trainer
from sol.metrics import Accuracy, Precision, Recall
from sol.model import Model


warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def collate_fn(dataset_items):
    result_batch = {}
    for k in dataset_items[0].keys():
        if k == "filename":
            result_batch[k] = [item[k] for item in dataset_items]
            continue
        result_batch[k] = torch.stack([torch.tensor(item[k]) for item in dataset_items])
    return result_batch


config = {
    "wandb_project": "cv_birds",
    "wandb_name": "2_lr_1e-5_resnet",
    "batch_size": 32
}


def train_classifier(train_gt: dict, train_images: str, fast_train: bool = False):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train_dataset = ImageDataset(train_images, type_set=Mode.TRAIN, gt=train_gt)
    val_dataset = ImageDataset(train_images, type_set=Mode.VAL, gt=train_gt)

    assert not len(set(train_dataset._idx) & set(val_dataset._idx))

    dataloaders = {
        "train": DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'], collate_fn=collate_fn),
        "val": DataLoader(val_dataset, shuffle=False, batch_size=config['batch_size'], collate_fn=collate_fn)
    }

    # build model architecture, then print to console
    model = Model(load_mobile=not fast_train)
    logger.info(model)
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    model = model.to(device)

    # get function handles of loss and metrics
    loss_module = nn.CrossEntropyLoss().to(device)

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, total_iters=1, factor=1
    )

    trainer = Trainer(
        model=model,
        metrics=[Accuracy("accuracy")],
        criterion=loss_module,
        optimizer=optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epochs=1000 if not fast_train else 1,
        log_step=10,
        len_epoch=100 if not fast_train else 3,
        lr_scheduler=lr_scheduler,
        fast_train=fast_train
    )

    trainer.train()


def classify(model_filename: str, test_image_dir: str) -> Dict[str, np.ndarray]:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logging.basicConfig(stream=sys.stdout, level=logging.WARN)

    model = Model()
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.to(device)
    model.eval()

    test_dataset = ImageDataset(test_image_dir, type_set=Mode.TEST)

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config['batch_size'],
        collate_fn=collate_fn,
        drop_last=False
    )

    results = {}
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch["img"])
            labels = torch.argmax(pred, dim=1).detach()
            results.update(dict(zip(batch["filename"], labels.cpu().numpy())))

    return results


from run import read_csv

if __name__ == "__main__":
    train_classifier(
        train_gt=read_csv("tests/00_test_img_input/train/gt.csv"),
        train_images="tests/00_test_img_input/train/images/",
        fast_train=False
    )

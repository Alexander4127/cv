import argparse
import collections
import logging
import warnings
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sol.model import Model
from sol.datasets import ImageDataset
from sol.trainer import Trainer
from sol.metrics import RealMSE

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
        result_batch[k] = torch.stack([torch.tensor(item[k]) for item in dataset_items])
    return result_batch


def main():
    config = {
        "wandb_project": "cv_faces",
        "wandb_name": "10_const_lr_resnet",
        "img_size": 128,
        "n_channels": 8,
        "batch_size": 32
    }

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    datasets = {
        "train": ImageDataset('tests/00_test_img_input/train', img_size=config['img_size'], is_train=True),
        "val": ImageDataset('tests/00_test_img_input/train', img_size=config['img_size'], is_train=False)
    }
    dataloaders = {
        "train": DataLoader(datasets["train"], shuffle=True, batch_size=config['batch_size'], collate_fn=collate_fn),
        "val": DataLoader(datasets["val"], shuffle=False, batch_size=config['batch_size'], collate_fn=collate_fn)
    }

    # build model architecture, then print to console
    model = Model(img_size=config['img_size'], n_channels=config['n_channels'])
    logger.info(model)

    # prepare for (multi-device) GPU training
    model = model.to(device)

    # get function handles of loss and metrics
    loss_module = nn.MSELoss().to(device)
    metrics = [RealMSE(name="real_mse")]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, total_iters=1, factor=1
    )

    trainer = Trainer(
        model=model,
        metrics=metrics,
        criterion=loss_module,
        optimizer=optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epochs=100,
        log_step=20,
        len_epoch=200,
        lr_scheduler=lr_scheduler
    )

    trainer.train()


if __name__ == "__main__":
    main()

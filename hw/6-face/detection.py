import argparse
import collections
import logging
import warnings
import pathlib
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sol.model import Model
from sol.datasets import ImageDataset, Mode
from sol.trainer import Trainer
from sol.utils import pred2coord

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
    "wandb_project": "cv_faces",
    "wandb_name": "8_resnet_full_wo_hor_aug",
    "img_size": 256,
    "n_channels": 8,
    "batch_size": 32
}


def train_detector(train_gt: dict, train_images: str, fast_train: bool = False):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    logging.basicConfig(stream=sys.stdout, level=logging.INFO if not fast_train else logging.INFO)

    train_dataset = ImageDataset(train_images, img_size=config['img_size'], type_set=Mode.TRAIN, gt=train_gt)
    val_dataset = ImageDataset(train_images, img_size=config['img_size'], type_set=Mode.VAL, gt=train_gt)

    assert not len(set(train_dataset._idx) & set(val_dataset._idx))

    dataloaders = {
        "train": DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'], collate_fn=collate_fn),
        "val": DataLoader(val_dataset, shuffle=False, batch_size=config['batch_size'], collate_fn=collate_fn)
    }

    # build model architecture, then print to console
    model = Model(img_size=config['img_size'], n_channels=config['n_channels'])
    logger.info(model)

    model = model.to(device)

    # get function handles of loss and metrics
    loss_module = nn.MSELoss().to(device)

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, total_iters=1, factor=1
    )

    trainer = Trainer(
        model=model,
        metrics=[],
        criterion=loss_module,
        optimizer=optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epochs=1000 if not fast_train else 1,
        log_step=10,
        len_epoch=100 if not fast_train else 10,
        lr_scheduler=lr_scheduler,
        fast_train=fast_train
    )

    trainer.train()


def detect(model_filename: str, test_image_dir: str) -> Dict[str, np.ndarray]:
    logging.basicConfig(stream=sys.stdout, level=logging.WARN)

    model = Model(img_size=config['img_size'], n_channels=config['n_channels'])
    model.load_state_dict(torch.load(model_filename))
    model.eval()

    test_dataset = ImageDataset(test_image_dir, img_size=config['img_size'], type_set=Mode.TEST)

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
            pred = pred2coord(model(batch["img"]), batch["size"]).numpy()
            results.update(dict(zip(batch["filename"], pred)))

    return results


from run import read_csv

if __name__ == "__main__":
    train_detector(
        train_gt=read_csv('tests/00_test_img_input/train/gt.csv'),
        train_images='tests/00_test_img_input/train/images',
        fast_train=False
    )

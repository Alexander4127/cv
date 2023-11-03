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
from torchvision.models import resnet18
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# --------------------------- utils ---------------------------


def pred2coord(pred: torch.Tensor, size: torch.Tensor, **kwargs):
    assert pred.shape[0] == size.shape[0] and size.shape[1] == 2

    coord = pred.detach().cpu().float().clone() + 0.5
    w = coord.shape[1]
    for idx, idx_range in enumerate([torch.arange(0, w, 2), torch.arange(1, w, 2)]):
        coord[:, idx_range] *= size[:, idx].unsqueeze(1)

    return coord


def coord2pred(ans: torch.Tensor, size: torch.Tensor, **kwargs):
    assert ans.shape[0] == size.shape[0] and size.shape[1] == 2

    pred = ans.cpu().float().clone()
    size = size.float()
    w = pred.shape[1]
    for idx, idx_range in enumerate([torch.arange(0, w, 2), torch.arange(1, w, 2)]):
        pred[:, idx_range] /= size[:, idx].unsqueeze(1)

    return pred - 0.5


# --------------------------- tracker ---------------------------

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()


# --------------------------- logger ---------------------------

class WanDBWriter:
    def __init__(self, config, logger):
        self.writer = None
        self.selected_module = ""

        try:
            import wandb
            wandb.login()

            wandb.init(
                project=config['wandb_project'],
                name=config['wandb_name']
            )
            self.wandb = wandb

        except ImportError:
            logger.warning("For use wandb install it via \n\t pip install wandb")

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def _scalar_name(self, scalar_name):
        return f"{scalar_name}_{self.mode}"

    def add_scalar(self, scalar_name, scalar):
        self.wandb.log({
            self._scalar_name(scalar_name): scalar,
        }, step=self.step)

    def add_scalars(self, tag, scalars):
        self.wandb.log({
            **{f"{scalar_name}_{tag}_{self.mode}": scalar for scalar_name, scalar in
               scalars.items()}
        }, step=self.step)

    def add_image(self, scalar_name, image):
        self.wandb.log({
            self._scalar_name(scalar_name): self.wandb.Image(image)
        }, step=self.step)

    def add_text(self, scalar_name, text):
        self.wandb.log({
            self._scalar_name(scalar_name): self.wandb.Html(text)
        }, step=self.step)

    def add_histogram(self, scalar_name, hist, bins=None):
        hist = hist.detach().cpu().numpy()
        np_hist = np.histogram(hist, bins=bins)
        if np_hist[0].shape[0] > 512:
            np_hist = np.histogram(hist, bins=512)

        hist = self.wandb.Histogram(
            np_histogram=np_hist
        )

        self.wandb.log({
            self._scalar_name(scalar_name): hist
        }, step=self.step)

    def add_table(self, table_name, table: pd.DataFrame):
        self.wandb.log({self._scalar_name(table_name): wandb.Table(dataframe=table)},
                       step=self.step)


# --------------------------- augments ---------------------------


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
        self._aug = TF.hflip

    def __call__(self, img: torch.Tensor, size: torch.Tensor, ans: torch.Tensor):
        ans = ans.clone()
        ans[::2] = size[0] - ans[::2]
        return self._aug(img), size, ans


class VerticalFlip(BaseAugmentation):
    def __init__(self):
        self._aug = TF.vflip

    def __call__(self, img: torch.Tensor, size: torch.Tensor, ans: torch.Tensor):
        ans = ans.clone()
        ans[0::2] = size[1] - ans[0::2]
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

# --------------------------- dataset ---------------------------


class Mode(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class ImageDataset(Dataset):
    def __init__(self,
                 image_dir: str,
                 img_size: int,
                 type_set: Mode,
                 gt: tp.Optional[dict] = None,
                 train_size: float = 0.8):
        self._image_dir = pathlib.Path(image_dir)
        self._transform = T.Compose([
            T.PILToTensor(),
            lambda t: t.float() / 255,
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._resize = T.Compose([
            T.Resize([img_size, img_size]),
            T.CenterCrop(224)
        ])

        if type_set == Mode.TRAIN:
            self._size_augments = SequentialAugmentation([
                # RandomApply(HorizontalFlip())
            ])
            self._st_augments = SequentialAugmentation([
                RandomApply(ColorJitter())
            ])

        if gt is None:
            self._idx = [name for name in os.listdir(image_dir) if name.endswith('.jpg')]
        else:
            self._idx = list(gt.keys())

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
            img = self._transform(image.convert('RGB'))

        assert img.shape[0] == 3 and len(img.shape) == 3
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


# --------------------------- model ---------------------------

# using model from torchvision

# --------------------------- trainer ---------------------------


TEST_IMAGE_SIZE = 100


class Trainer:
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            metrics,
            criterion,
            optimizer,
            config,
            device,
            dataloaders,
            epochs,
            log_step,
            len_epoch,
            lr_scheduler,
            fast_train=True
    ):
        self.device = device
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer

        self.train_dataloader = dataloaders["train"]
        self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = log_step
        self.fast_train = fast_train
        self.writer = WanDBWriter(config, metrics) if not self.fast_train else None
        self.name = config['wandb_name']
        self.val_loss = 1e9

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )

        self.start_epoch = 1
        self.epochs = epochs

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["img", "ans"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        if not self.fast_train:
            self.writer.add_scalar("epoch", epoch)
        last_train_metrics = {}
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            batch = self.process_batch(
                batch,
                is_train=True,
                metrics=self.train_metrics,
            )
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0 and not self.fast_train:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                self._log_image(**batch)

                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics
        if self.fast_train:
            return log

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()

        batch["pred"] = self.model(batch["img"])
        batch["loss"] = self.criterion(batch["pred"], batch["ans"]) * TEST_IMAGE_SIZE**2

        if is_train:
            batch["loss"].backward()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        metrics.update("loss", batch["loss"].item())
        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            if not self.fast_train:
                self.writer.set_step(epoch * self.len_epoch, part)
                self._log_image(**batch)
                self._log_scalars(self.evaluation_metrics)

        if self.fast_train:
            return self.evaluation_metrics.result()

        if self.evaluation_metrics.avg("loss") < self.val_loss:
            self.val_loss = self.evaluation_metrics.avg("loss")
            self._save_model()

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        return self.evaluation_metrics.result()

    def _save_model(self):
        logger.info(f'    Saving model to models/{self.name}...')
        torch.save(self.model.state_dict(), f'models/{self.name}')

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _log_image(self, img, pred, size, ans, **kwargs):
        bs = len(ans)
        idx = np.random.choice(np.arange(len(img)))
        plt.imshow(np.clip(np.transpose(img[idx].detach().cpu().numpy(), axes=(1, 2, 0)), 0, 1))

        img_size = torch.ones([bs, 2]) * self.config['img_size']
        pred_coord = pred2coord(pred, img_size).detach().cpu().numpy()
        real_img_coord = pred2coord(ans, img_size).numpy()

        x_idx, y_idx = np.arange(pred.shape[1]) % 2 == 0, np.arange(pred.shape[1]) % 2 == 1
        plt.scatter(pred_coord[idx, x_idx], pred_coord[idx, y_idx], label='Pred')
        plt.scatter(real_img_coord[idx, x_idx], real_img_coord[idx, y_idx], label='Real')
        plt.legend()
        plt.savefig(f'img/{self.name}.png')
        plt.close()
        with Image.open(f'img/{self.name}.png') as image:
            self.writer.add_image('Pred and real', image)

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged information to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))


# --------------------------- detection ---------------------------


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
    "wandb_name": "8_real_resnet",
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
    # model = Model(img_size=config['img_size'], n_channels=config['n_channels'])
    model = resnet18(num_classes=28)
    logger.info(model)

    model = model.to(device)

    # get function handles of loss and metrics
    loss_module = nn.MSELoss().to(device)

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
        len_epoch=100 if not fast_train else 3,
        lr_scheduler=lr_scheduler,
        fast_train=fast_train
    )

    trainer.train()


def detect(model_filename: str, test_image_dir: str) -> Dict[str, np.ndarray]:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logging.basicConfig(stream=sys.stdout, level=logging.WARN)

    # model = Model(img_size=config['img_size'], n_channels=config['n_channels'])
    model = resnet18(num_classes=28)
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.to(device)
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

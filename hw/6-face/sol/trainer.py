import random
from pathlib import Path
from random import shuffle
import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from sol.tracker import MetricTracker
from sol.logger import WanDBWriter
from sol.utils import pred2coord, coord2pred


logger = logging.getLogger(__name__)


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
        self.writer = WanDBWriter(config, metrics)

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
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            batch = self.process_batch(
                batch,
                is_train=True,
                metrics=self.train_metrics,
            )
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
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

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()

        batch["pred"] = self.model(batch["img"])
        batch["loss"] = self.criterion(batch["pred"], batch["ans"])

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
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_image(**batch)
            self._log_scalars(self.evaluation_metrics)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

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

    def _log_image(self, img, pred, size, real_ans, **kwargs):
        bs = len(real_ans)
        idx = np.random.choice(np.arange(len(img)))
        plt.imshow(np.transpose(img[idx].detach().cpu().numpy().astype(np.uint8), axes=(1, 2, 0)))

        img_size = torch.ones([bs, 2]) * self.model.img_size
        pred_coord = pred2coord(pred, img_size).detach().cpu().numpy()
        real_norm = coord2pred(real_ans, size)
        real_img_coord = pred2coord(real_norm, img_size)

        x_idx, y_idx = np.arange(pred.shape[1]) % 2 == 0, np.arange(pred.shape[1]) % 2 == 1
        plt.scatter(pred_coord[idx, x_idx], pred_coord[idx, y_idx], label='Pred')
        plt.scatter(real_img_coord[idx, x_idx], real_img_coord[idx, y_idx], label='Real')
        plt.legend()
        plt.savefig('1.png')
        plt.close()
        with Image.open('1.png') as image:
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

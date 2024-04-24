#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
import os
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb import Histogram

from manifold_experiments.constants import HEADS, OPTIMIZERS
from manifold_experiments.datasets import VideoDataset
from manifold_experiments.manifold_models import (
    MMCRTwoStageTwoHeadPredictor,
    ProjectionHead,
    TwoStageTwoHeadPredictor,
    create_encoder,
)
from manifold_experiments.utils.curvature import manifold_curvature
from manifold_experiments.utils.flatten import flatten
from manifold_experiments.utils.knn import manifold_knn
from manifold_experiments.utils.least_square_regression import manifold_lstsq
from manifold_experiments.utils.manifold_statistics import (
    extract_activations,
    get_feature_extractor,
    make_manifold_data,
    manifold_analysis,
)


class LogJobReturnCallback(Callback):
    """Log the job's return value or error upon job end"""

    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        if job_return.status == JobStatus.COMPLETED:
            self.log.info(f"Succeeded with return value: {job_return.return_value}")
        elif job_return.status == JobStatus.FAILED:
            self.log.error("", exc_info=job_return._return_value)
        else:
            self.log.error("Status unknown. This should never happen.")


@hydra.main(config_path="configs", config_name="mmcr_train", version_base="1.1")
def train(cfg: OmegaConf):
    print(OmegaConf.to_yaml(cfg), flush=True)
    set_seed(cfg.seed)

    accelerator = Accelerator(
        cpu=cfg.cpu,
        log_with="wandb",
        project_dir=HydraConfig.get().runtime.output_dir,
        mixed_precision=cfg.mixed_precision,
    )

    run_name = (
        f"{cfg.model.name}_{cfg.dataset.train_name}_loss_{cfg.model.manifold_loss}"
    )
    run_name += f"_lr_{cfg.learning_rate}_batch_{cfg.batch_size}_seed_{cfg.seed}"

    accelerator.init_trackers(
        project_name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, enum_to_str=True),
        init_kwargs={
            "wandb": {
                "name": hydra.core.hydra_config.HydraConfig.get().job.override_dirname
            }
        },
    )

    encoder = create_encoder(
        model_name=cfg.model.encoder.type,
        weights=cfg.model.encoder.weights,
        **cfg.model.encoder.kwargs,
    )

    model_type = (
        TwoStageTwoHeadPredictor
        if cfg.model.type == "default"
        else MMCRTwoStageTwoHeadPredictor
    )

    model = model_type(
        encoder=encoder,
        linear_head=HEADS[cfg.model.classification_type](
            **cfg.model.classification_kwargs
        )
        if OmegaConf.select(cfg, "model.classification_type")
        else None,
        autoregressive_head=HEADS[cfg.model.autoregressive_type](
            **cfg.model.autoregressive_kwargs
        )
        if OmegaConf.select(cfg, "model.autoregressive_type")
        else None,
        projector=ProjectionHead(**cfg.model.projection_kwargs)
        if OmegaConf.select(cfg, "model.projection_kwargs")
        else nn.Identity(),
        norm=OmegaConf.select(cfg, "model.norm"),
        manifold_loss=OmegaConf.select(cfg, "model.manifold_loss", default="capacity"),
    )

    # Freeze layers up until the specified layer, if present
    if OmegaConf.select(cfg, "model.freeze_until"):
        for name, param in model.named_parameters():
            if cfg.model.freeze_until != "all" and cfg.model.freeze_until in name:
                break
            param.requires_grad = False

    # model = model.to(accelerator.device)

    # Load dataset
    train_dataset = VideoDataset(
        name=cfg.dataset.train_name,
        frames_per_clip=cfg.dataset.frames_per_clip,
        **cfg.dataset.train_kwargs,
    )
    val_dataset = VideoDataset(
        name=cfg.dataset.val_name,
        frames_per_clip=cfg.dataset.frames_per_clip,
        **cfg.dataset.val_kwargs,
    )
    manifold_train_dataset = VideoDataset(
        name=cfg.manifold.train_dataset,
        **cfg.manifold.train_dataset_kwargs,
    )
    manifold_val_dataset = VideoDataset(
        name=cfg.manifold.val_dataset,
        **cfg.manifold.val_dataset_kwargs,
    )
    sampled_manifold_dataset = make_manifold_data(
        manifold_train_dataset,
        cfg.manifold.sampled_classes,
        cfg.manifold.examples_per_class,
    )

    # Load dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    # Load optimizer and scheduler
    optimizer = OPTIMIZERS[cfg.optimizer.type](
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        **cfg.optimizer.kwargs,
    )

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    for epoch in range(cfg.epochs):
        # Analyze manifold
        if epoch % cfg.manifold.calculate_every == 0:
            manifold_statistics = {}
            with torch.no_grad():
                if "ellipsoid" in cfg.manifold.calculate:
                    feature_extractor = get_feature_extractor(
                        model.encoder, cfg.manifold.return_nodes
                    )
                    activations = extract_activations(
                        sampled_manifold_dataset, feature_extractor
                    )
                    analysis = manifold_analysis(activations, cfg.manifold.calculate)
                    manifold_statistics.update(analysis)

                if "knn" in cfg.manifold.calculate:
                    knn_results = manifold_knn(
                        model,
                        manifold_train_dataset,
                        manifold_val_dataset,
                    )
                    manifold_statistics.update(knn_results)

                # NOTE: This is only meaningful when performed on video datasets,
                # not on MNIST
                if "lstsq" in cfg.manifold.calculate:
                    lstsq_results = manifold_lstsq(model, val_dataset)
                    manifold_statistics.update(lstsq_results)

                if "curvature" in cfg.manifold.calculate:
                    curvature_results = manifold_curvature(model, val_dataset)
                    manifold_statistics.update(curvature_results)

        # Train
        train_loss = 0
        model.train()
        for data, label in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)
            losses = model.calculate_loss(data, label)
            assert type(losses) is dict, "Losses must be returned in a dictionary"

            loss = losses[model.manifold_loss]
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        val_losses = dict()
        model.eval()
        with torch.no_grad():
            for data, label in tqdm(val_dataloader, desc=f"Val Epoch {epoch}"):
                losses = model.calculate_loss(data, label)
                assert type(losses) is dict, "Losses must be returned in a dictionary"
                if len(val_losses) == 0:
                    for key, val in losses.items():
                        val_losses[key] = [val.item()]
                else:
                    for key, val in losses.items():
                        val_losses[key] += [val.item()]
        val_losses = {key: np.mean(val) for key, val in val_losses.items()}

        # Log to tensorboard
        log_obj = {
            "train_loss": train_loss / len(train_dataloader),
            "val_loss": val_losses[model.manifold_loss],
            "epoch": epoch,
            **val_losses,
            **manifold_statistics,
        }

        log_obj = flatten(log_obj)

        accelerator.log(
            {
                key: value
                for key, value in log_obj.items()
                if ("singular_values" not in key)
            },
            step=epoch,
        )

        accelerator.print(
            f"Epoch {epoch}: Train Loss {log_obj['train_loss']}, Val Loss {log_obj['val_loss']}"
        )

    os.makedirs(
        os.path.join(
            HydraConfig.get().runtime.output_dir,
            str(HydraConfig.get().job.override_dirname),
        ),
        exist_ok=True,
    )

    accelerator.save(
        model.state_dict(),
        os.path.join(
            HydraConfig.get().runtime.output_dir,
            str(HydraConfig.get().job.override_dirname),
            "model.pth",
        ),
    )
    torch.save(
        model.encoder.state_dict(),
        os.path.join(
            HydraConfig.get().runtime.output_dir,
            str(HydraConfig.get().job.override_dirname),
            "encoder.pth",
        ),
    )
    accelerator.end_training()


if __name__ == "__main__":
    train()

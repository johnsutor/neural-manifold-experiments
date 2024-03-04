#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from aim import Distribution
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import HEADS, OPTIMIZERS
from datasets import VideoDataset
from models import (
    MMCRTwoStageTwoHeadPredictor,
    TwoStageTwoHeadPredictor,
    create_encoder,
)
from utils.flatten import flatten
from utils.knn import manifold_knn
from utils.manifold_statistics import (
    extract_activations,
    get_feature_extractor,
    make_manifold_data,
    manifold_analysis,
)


@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def train(cfg: OmegaConf):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    accelerator = Accelerator(
        cpu=cfg.cpu,
        log_with="aim",
        project_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        mixed_precision=cfg.mixed_precision,
    )

    accelerator.init_trackers(
        project_name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, enum_to_str=True),
        init_kwargs={},
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
        norm=OmegaConf.select(cfg, "cfg.model.norm"),
    )

    # Freeze layers up until the specified layer, if present
    if OmegaConf.select(cfg, "model.freeze_until"):
        for name, param in model.named_parameters():
            if cfg.model.freeze_until != "all" and cfg.model.freeze_until in name:
                break
            param.requires_grad = False

    model = model.to(accelerator.device)

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
            with torch.no_grad():
                manifold_statistics = {}
                feature_extractor = get_feature_extractor(
                    model.encoder, cfg.manifold.return_nodes
                )
                activations = extract_activations(
                    sampled_manifold_dataset, feature_extractor
                )
                if "knn" in cfg.manifold.calculate:
                    knn_results = manifold_knn(
                        model,
                        manifold_train_dataset,
                        manifold_val_dataset,
                    )
                    manifold_statistics.update(knn_results)

                analysis = manifold_analysis(activations, cfg.manifold.calculate)
                manifold_statistics.update(analysis)

        # Train
        train_loss = 0
        model.train()
        for data, label in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)
            loss = model.calculate_loss(data, label)
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for data, label in tqdm(val_dataloader, desc=f"Val Epoch {epoch}"):
                loss = model.calculate_loss(data, label)
                val_loss += loss.item()

        # Log to tensorboard
        log_obj = {
            "train_loss": train_loss / len(train_dataloader),
            "val_loss": val_loss / len(val_dataloader),
            "epoch": epoch,
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

        for key, value in log_obj.items():
            if "singular_values" in key and value is not None:
                accelerator.get_tracker("aim").tracker.track(
                    Distribution(value), name=key, epoch=epoch
                )

        accelerator.print(
            f"Epoch {epoch}: Train Loss {log_obj['train_loss']}, Val Loss {log_obj['val_loss']}"
        )

    accelerator.save(model.state_dict(), "model.pth")
    torch.save(
        model.encoder.state_dict(),
        os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "encoder.pth"
        ),
    )


if __name__ == "__main__":
    train()

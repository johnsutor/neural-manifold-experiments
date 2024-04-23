import os
import warnings
from typing import List, Literal, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model, list_models
from torch.linalg import svdvals


class ProjectionHead(nn.Module):
    def __init__(self, features: List[int]):
        super().__init__()
        layers = []
        for i in range(len(features) - 1):
            layers.append(nn.Linear(features[i], features[i + 1]))
            layers.append(nn.ReLU())

        # Don't include last ReLU
        self.model = nn.Sequential(*layers[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def create_encoder(
    model_name: str,
    weights: os.PathLike = None,
    **kwargs,
) -> nn.Module:
    """
    Create an encoder model.

    Args:
        model_name : str
            Model name. The model should be available in
            [Torch Image Models](https://github.com/huggingface/pytorch-image-models)
        weights : os.PathLike, optional
            Path to the weights file, by default None
        **kwargs
            Additional keyword arguments for the model
    Returns:
        nn.Module
            Encoder model
    """
    assert model_name in list_models(), f"Model {model_name} not available in timm"
    model = create_model(
        model_name,
        **kwargs,
    )
    if weights is not None:
        model.load_state_dict(torch.load(weights))

    return model


class TwoStageTwoHeadPredictor(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        linear_head: Optional[nn.Module],
        autoregressive_head: Optional[nn.Module],
        projector: nn.Module = nn.Identity(),
        **kwargs,
    ):
        """
        Two-stage predictor model.

        Args:
        encoder : nn.Module
            Encoder module. The encoder should have the method
            forward_features() which returns the encoded image,
            as opposed to forward which returns the linear outputs
            (for compatibility with
            [Torch Image Models](https://github.com/huggingface/pytorch-image-models))
        linear_head : nn.Module
            Module for transforming encoded features.
        autoregressive_head : nn.Module
            Module for transforming encoded features.
        projector : nn.Module, optional
            Projector module. The default is nn.Identity().
        """

        super().__init__()
        self.encoder = encoder
        self.linear_head = linear_head
        self.autoregressive_head = autoregressive_head
        self.projector = projector

        if "norm" in kwargs:
            # Warn user they should use MMCR instead
            warnings.warn(
                "norm parameter not used in TwoStageTwoHeadPredictor. Use MMCRTwoStageTwoHead instead."
            )

    def get_latent(self, x) -> List[torch.Tensor]:
        B, T, *_ = x.shape

        x = [self.encoder(x[:, i, :, :, :]) for i in range(T)]
        x = torch.stack(x, dim=1)
        x = x.reshape(B, T, -1)

        return x

    def get_projection(self, x: torch.Tensor) -> torch.Tensor:

        latents = self.get_latent(x)
        latents = torch.stack(
            [self.projector(latent) for latent in latents.unbind(dim=0)]
        )

        return latents

    def calculate_loss(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        latents = self.get_latent(input)
        B, T, *_ = latents.shape

        linear_loss = 0.0
        if self.linear_head is not None:
            linear_pred = self.linear_head(latents.reshape(B * T, -1))
            linear_loss = F.cross_entropy(linear_pred, label.reshape(B * T))

        autoregressive_loss = 0.0
        if self.autoregressive_head is not None:
            autoregressive_pred, *_ = self.autoregressive_head(latents)
            autoregressive_loss = F.mse_loss(
                autoregressive_pred[:, -2, :], latents[:, -1, :]
            )

        return linear_loss + autoregressive_loss


class MMCRTwoStageTwoHeadPredictor(TwoStageTwoHeadPredictor):
    def __init__(
        self,
        encoder: nn.Module,
        linear_head: Optional[nn.Module],
        autoregressive_head: Optional[nn.Module],
        projector: nn.Module = nn.Identity(),
        norm: Literal[0, 1] = 0,
        manifold_loss: Literal["capacity", "radius", "dimensionality",
                               "capacity_obj", "radius_obj", "dimensionality_obj"] = "capacity",
    ):
        super().__init__(encoder, None, autoregressive_head, projector)
        self.norm = norm
        self.manifold_loss = manifold_loss

    def calculate_loss(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # encode the frames (treat each frame as a separate sample)
        latents = self.get_projection(input)

        assert latents.dim() == 3, \
            f"Latents must have shape (batch, frame, features): shape is {latents.shape}"

        latents = F.normalize(latents, p=2, dim=-1)
        centroid = latents.mean(dim=1)

        # SVD of centroids - shape (min(batch, feature))
        S_c = svdvals(centroid)
        # SVD of latents - shape (batch, min(frame, feature))
        S_z = svdvals(latents)

        if self.norm == 0:
            S_c = S_c.tanh()
            S_z = S_z.tanh()

        S_c = S_c.abs() / math.sqrt(max(centroid.shape))
        S_z = S_z.abs() / math.sqrt(max(latents.shape[1:]))

        # Original from MMCR Paper: -S_c.sum() + lambda *  S_l.sum() / latents.shape[0]
        # However, implicit manifold compression actually reduces the mean augmentation
        # manifold nuclear norm, so the S_l term isn't necessary.

        # Get Radius, Dimension and Capacity from Singular Values
        rad_c = (S_c**2).sum().sqrt()
        dim_c = S_c.sum()**2 / rad_c**2 / S_c.shape[-1]
        alpha_c = rad_c * dim_c.sqrt()

        rad_z = (S_z**2).sum(-1).sqrt()
        dim_z = S_z.sum(-1)**2 / (S_z**2).sum(-1) / S_z.shape[-1]
        alpha_z = rad_z * dim_z.sqrt()

        losses = dict(capacity=-alpha_c,
                      radius=-rad_c,
                      dimensionality=-dim_c,
                      capacity_obj=alpha_z.mean(),
                      radius_obj=rad_z.mean(),
                      dimensionality_obj=dim_z.mean()
                      )

        return losses

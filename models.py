import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model, list_models


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

    def get_latent(self, x) -> List[torch.Tensor]:
        # input is shape (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.encoder.forward(x)
        x = x.reshape(B, T, -1)
        return x

    def calculate_loss(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        latents = self.get_latent(input)
        B, T, C = latents.shape

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


class TwoStagePredictor(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        projector: nn.Module = nn.Identity(),
    ):
        """
        Two-stage predictor model.
        Parameters
        ----------
        encoder : nn.Module
            Encoder module. The encoder should have the method
            forward_features() which returns the encoded image,
            as opposed to forward which returns the linear outputs
            (for compatibility with
            [Torch Image Models](https://github.com/huggingface/pytorch-image-models))
        head : nn.Module
            Module for transforming encoded features.
        projector : nn.Module, optional
            Projector module. The default is nn.Identity().

        """
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.projector = projector

    def get_latent(self, x) -> List[torch.Tensor]:
        return self.encoder.forward(x)

    def get_regressed(self, x, hidden) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.encoder.forward(x)
        encoded_image = latents[-1].reshape(latents[-1].shape[0], -1)
        if isinstance(self.head, nn.Linear):
            latent, hidden = self.head(encoded_image), None
        else:
            latent, hidden = self.head(encoded_image, hidden)
        return latent, hidden

    def get_decoded(self, latent) -> torch.Tensor:
        return self.projector(latent)

    def forward(
        self, x, hidden=None
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        latents = self.get_latent(x)
        pred_latent, hidden = self.get_regressed(x, hidden)
        pred_latent = self.get_decoded(pred_latent)
        return latents, pred_latent, hidden

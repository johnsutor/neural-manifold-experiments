import os
import warnings
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model, list_models
from torch.linalg import svdvals


class ProjectionHead(nn.Module):
    def __init__(self, in_features: int, hidden_features: List[int], out_features: int):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_features[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_features) - 1):
            layers.append(nn.Linear(hidden_features[i], hidden_features[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_features[-1], out_features))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


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
    ):
        super().__init__(encoder, None, autoregressive_head, projector)
        self.norm = norm

    def calculate_loss(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # encode the frames (treat each frame as a separate sample)
        latents = self.get_latent(input)
        latents = torch.stack(
            [self.projector(latent) for latent in latents.unbind(dim=0)]
        )

        assert (
            latents.dim() == 3
        ), f"Latent shape is not correct: shape is {latents.shape}"

        latents = F.normalize(latents, p=2, dim=-1)

        centroid = latents.mean(dim=1)

        # SVD
        S_c = svdvals(centroid)
        S_l = svdvals(latents)

        # TODO: Try softmax
        if self.norm == 0:
            S_c, S_l = S_c.tanh(), S_l.tanh()

        return -S_c.sum() + S_l.sum() / latents.shape[0]

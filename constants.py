from functools import partial

import torch
from timm.optim import Lars
from torchvision.transforms.v2 import (
    Compose,
    Lambda,
    Normalize,
    RandomCrop,
    Resize,
    ToDtype,
    ToImage,
)

VIDEO_DATASETS = ["kinetics", "physion", "moving_mnist"]
IMAGE_DATASETS = ["mnist"]

TASKS = ["autoregression", "classification", "ssl"]

HEADS = {
    "lstm": partial(torch.nn.LSTM, batch_first=True),
    "rnn": partial(torch.nn.RNN, batch_first=True),
    "identity": torch.nn.Identity,
    "linear": torch.nn.Linear,
    "flatten": torch.nn.Flatten,
}

OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "lars": Lars}

SCHEDULERS = {
    "step": torch.optim.lr_scheduler.StepLR,
    "linear_decay": torch.optim.lr_scheduler.LinearLR,
}

TRANSFORMS = {
    "mnist": Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            RandomCrop((64, 64), padding=36),
            Normalize(mean=[0.5], std=[0.5]),
            Lambda(lambda x: x.unsqueeze(0)),
        ]
    ),
    "moving_mnist": Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize((64, 64), antialias=True),
            Normalize(mean=[0.5], std=[0.5]),
        ]
    ),
    "kinetics": Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize((64, 64), antialias=True),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
    "physion": Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize((64, 64), antialias=True),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
    "custom_moving_mnist": Compose(
        [
            ToDtype(torch.float32, scale=True),
            Resize((64, 64), antialias=True),
            Normalize(mean=[0.5], std=[0.5]),
        ]
    ),
}

TARGET_TRANSFORMS = {
    "mnist": Lambda(lambda x: torch.tensor(x, dtype=torch.long).unsqueeze(0)),
    "moving_mnist": None,
    "kinetics": None,
    "physion": None,
    "custom_moving_mnist": None,
}

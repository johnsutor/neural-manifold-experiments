import os
from typing import Tuple

import h5py
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, Kinetics, MovingMNIST

from manifold_experiments.constants import TARGET_TRANSFORMS, TRANSFORMS
from manifold_experiments.physion import PhysionAllDataset


class CustomMovingMnistPyTorchDataset(Dataset):
    def __init__(self, path: str, train: bool = True, **kwargs):
        self.path = path
        self.train = train
        with h5py.File(self.path, "r") as hf:
            self.frames = hf["frames"][:]
            self.digits = hf["digits"][:]
            self.angles = hf["angles"][:]
            self.xs = hf["xs"][:]
            self.ys = hf["ys"][:]
            self.speeds = hf["speeds"][:]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        """Returns the video frames, the digit, the angle,
        the x position, the y position, and the speed (in that order)"""

        # Return first ten if train, last ten if test
        if self.train:
            return (
                self.frames[idx][:10],
                self.digits[idx][:10],
                self.angles[idx][:10],
                self.xs[idx][:10],
                self.ys[idx][:10],
                self.speeds[idx][:10],
            )

        else:
            return (
                self.frames[idx][10:],
                self.digits[idx][10:],
                self.angles[idx][10:],
                self.xs[idx][10:],
                self.ys[idx][10:],
                self.speeds[idx][10:],
            )


class VideoDataset(Dataset):
    """
    Video dataset wrapper
    """

    def __init__(
        self,
        root: os.PathLike,
        name: str = "moving_mnist",
        frames_per_clip: int = 7,
        verbose: bool = True,
        stride_range: Tuple[int, int] = (1, 1),
        **kwargs,
    ):
        """
        Parameters
        ----------
        root : str
            Path to the root of the dataset.  If None, downloads to default path "./data".
            The default is None.
        name: str = "moving_mnist"
            Dataset to use. The default is "moving_mnist".
        frames_per_clip : int, optional
            Number of frames to include in the context window. The default is 7.
        transform : callable, optional
            Optional transform to be applied on a sample.
        verbose : bool, optional
            Whether to print information about the dataset.
        stride_range: tuple[int, int] =  (1,1)
            Stride to use when sampling frames. It will be randomly selected from the
            range provided by the tuple, where the min is the first element and the max
            is the last element (similar to the ngram_range argument in the CountVectorizer
            in scikit-learn). The default is (1,1).
        """

        self.root = root
        self.name = name
        self.frames_per_clip = frames_per_clip
        self.verbose = verbose
        self.stride_range = stride_range
        self.kwargs = kwargs
        self.transform = TRANSFORMS.get(self.name, None)
        self.target_transform = TARGET_TRANSFORMS.get(self.name, None)

        assert (
            self.stride_range[0] <= self.stride_range[1]
        ), "Invalid stride range specified. The first element must be less than or equal to the second element."

        if self.name == "moving_mnist":
            self.data = MovingMNIST(root=self.root, **self.kwargs)
        elif self.name == "physion":
            self.data = PhysionAllDataset(
                root=self.root,
                frames_per_clip=frames_per_clip,
                stride_range=stride_range,
                **self.kwargs,
            )
        elif self.name == "mnist":
            self.data = MNIST(root=self.root, **self.kwargs)
            self.frames_per_clip = 1
        elif self.name == "kinetics":
            self.data = Kinetics(
                root=self.root,
                **self.kwargs,
            )
        elif self.name == "custom_moving_mnist":
            self.data = CustomMovingMnistPyTorchDataset(self.root, **self.kwargs)
        else:
            raise NotImplementedError(f"Dataset {self.name} not implemented")

        if self.name not in [
            "mnist",
            "physion",
            "custom_moving_mnist",
        ]:  # TODO: check custom moving mnist
            self._assert_stride_ok()

    def _assert_stride_ok(self):
        """
        Checks that the maximum stride is OK for the given dataset
        and desired number of frames returned per clip.
        """
        for _ in range(10):  # Check 10 random samples
            frames = self.data[torch.randint(0, len(self.data), (1,)).item()]

            # Check if includes label
            if len(frames) == 2:
                frames = frames[0]

            frames = frames[
                0 : self.stride_range[1] * self.frames_per_clip : self.stride_range[1]
            ]

            assert (
                frames.shape[0] == self.frames_per_clip
            ), "The maximum stride provided is too large"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.name in ["moving_mnist", "kinetics"]:
            frames = self.data[idx]
        elif self.name == "custom_moving_mnist":
            frames, labels, angles, xs, ys, speeds = self.data[idx]
            frames = torch.from_numpy(frames).float().unsqueeze(1)
        else:
            frames, labels = self.data[idx]

        if self.transform:
            frames = self.transform(frames)

        video_len = len(frames)
        frames_per_clip = self.frames_per_clip

        assert video_len >= frames_per_clip, "videos are shorter than clip lengths"

        if video_len > frames_per_clip:
            min_stride = max(self.stride_range[0], 1)
            max_stride = min(self.stride_range[1], video_len // frames_per_clip)

            stride = max_stride
            if min_stride < max_stride:
                stride = torch.randint(min_stride, max_stride, (1,)).item()

            start = torch.randint(0, video_len - stride * frames_per_clip, (1,)).item()
            stop = start + stride * frames_per_clip
            frames = frames[start:stop:stride]

        if self.name in ["moving_mnist", "kinetics"]:
            labels = torch.zeros(frames.shape[0], dtype=torch.long)

        if self.target_transform:
            labels = self.target_transform(labels)

        return frames, labels


if __name__ == "__main__":
    # Test that each dataset can be loaded
    for dataset in [
        "mnist",
        "moving_mnist",
        "kinetics",
        "physion",
        "custom_moving_mnist",
    ]:
        dataset = VideoDataset(root="./data", dataset=dataset)
        x, y = dataset[0]
        assert x.dim() == 4
        assert y.dim() == 2

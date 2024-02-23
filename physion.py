import io
import os
import shutil
import tarfile
from typing import Any, Callable, List, Union

import h5py
import requests
import torch
from decord import VideoReader, bridge, cpu
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class PhysionTrainDynamics(Dataset):
    """
    Dataset for training dynamics model on Physion dataset.
    It includes all of the MP4 files in the PhysionTrainMP4s
    at 30FPS.
    """

    def __init__(
        self,
        root: os.PathLike,
        frames_per_clip: int = 7,
        task: str = "readout",
        verbose: bool = True,
        stride_range: tuple[int, int] = (1, 1),
    ):
        """
        Parameters
        ----------
        root : str
            Root directory of the dataset.
        task: str = "readout"
            Task of the dataset, either "readout" or "training". The default is "readout".
        frames_per_clip : int, optional
            Number of frames to include in the context window. The default is 7.
        transform : callable, optional
            Optional transform to be applied on a sample.
        verbose : bool, optional
            Whether to print information about the dataset.

        """
        bridge.set_bridge("torch")

        self.valid_tasks = ["readout", "training"]
        task = task.lower()

        assert task in self.valid_tasks, (
            "Invalid task specified. Valid tasks are: "
            + str(",".join(self.valid_tasks))
        )

        self.root = os.path.join(os.path.abspath(root), "PhysionTrainMP4s")
        self.task = task
        self.verbose = verbose
        self.frames_per_clip = frames_per_clip
        self.stride_range = stride_range

        self.path = os.path.join(self.root, self.task)

        if not os.path.exists(self.path) or not len(os.listdir(self.path)):
            self._download_files()

    def _download_files(self):
        # Only download if the root directory does not exist
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)

            url = "https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/PhysionTrainMP4s.tar.gz"

            response = requests.get(url, stream=True)

            if response.status_code == 200:
                file = tarfile.open(fileobj=response.raw, mode="r|gz")
                file.extractall(path=self.root)

        for task in self.valid_tasks:
            if (
                not os.path.exists(os.path.join(self.root, task))
                or len(os.listdir(os.path.join(self.root, task))) == 0
            ):
                os.makedirs(os.path.join(self.root, task), exist_ok=True)
                task_folders = [
                    f
                    for f in os.listdir(os.path.join(self.root, "PhysionTrainMP4s"))
                    if task in f
                ]
                print(task_folders)
                for folder in task_folders:
                    for file in os.listdir(
                        os.path.join(self.root, "PhysionTrainMP4s", folder)
                    ):
                        shutil.copy(
                            os.path.join(self.root, "PhysionTrainMP4s", folder, file),
                            os.path.join(self.root, task, file),
                        )

        # Remove the original folder
        shutil.rmtree(os.path.join(self.root, "PhysionTrainMP4s"))

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        files = os.listdir(self.path)
        file = os.path.join(self.path, files[idx])

        vr = VideoReader(file, ctx=cpu(0))

        video_len = len(len(vr))
        frames_per_clip = self.frames_per_clip

        min_stride = max(self.stride_range[0], 1)
        max_stride = min(self.stride_range[1], video_len // frames_per_clip)

        if min_stride >= max_stride:
            stride = max_stride
        else:
            stride = torch.randint(min_stride, max_stride, (1,)).item()

        start = torch.randint(0, video_len - stride * frames_per_clip, (1,)).item()
        stop = start + stride * frames_per_clip

        # frames = vr.get_batch(range(0, len(vr))).permute(0, 3, 1, 2)
        # frames = vr.get_batch(range(0, 30)).permute(0, 3, 1, 2)
        # frames = torchvision.io.read_video(file, output_format="TCHW", pts_unit='sec')[0]

        frames = vr.get_batch(range(start, stop, stride)).permute(0, 3, 1, 2)

        return frames


class PhysionAllDataset(Dataset):
    """
    PyTorch Wrapper Physion dataset. This dataset contains
    metadata about each of the experiments conducted. This
    class will download the dataset at the given path if it
    does not exist.
    """

    def __init__(
        self,
        # We require specifying a path (changed to root to be consistent with Kinetics400)
        root: os.PathLike,
        split: str,  # Let's call it split to be consistent with Kinetics400
        scenarios: Union[List[str], str],
        frames_per_clip: int = 7,
        transform: Union[Callable[[Any], torch.Tensor], None] = None,
        verbose: bool = True,
        stride_range: tuple[int, int] = (1, 1),
    ):
        """
        Parameters
        ----------
        scenarios : str
            Name of the scenario.
        split : str
            Type of the dataset.
        root : str
            Path to the dataset.  If None, downloads to default root "./data".
            The default is None.
        frames_per_clip : int, optional
            Number of frames to include in the context window. The default is 7.
        transform : callable, optional
            Optional transform to be applied on a sample.
        verbose : bool, optional
            Whether to print information about the dataset.
        """
        self.valid_scenarios = [
            "Dominoes",
            "Support",
            "Collide",
            "Contain",
            "Drop",
            "Link",
            "Roll",
            "Drape",
        ]
        self.valid_splits = ["dynamics_training", "readout_training", "readout_test"]

        if not isinstance(scenarios, list):
            if scenarios == "all":
                scenarios = self.valid_scenarios
            else:
                scenarios = [scenarios]

        assert (
            set(scenarios) <= set(self.valid_scenarios) and len(scenarios) != 0
        ), f"Invalid scenario(s) specified. Valid scenarios are: {self.valid_scenarios}"
        assert (
            split in self.valid_splits
        ), f"Invalid split(s) specified. Valid splits are: {self.valid_splits}"

        self.root = os.path.join(os.path.abspath(root), "physion")

        self.scenarios = scenarios
        self.split = split
        self.transform = transform
        self.verbose = verbose
        self.frames_per_clip = frames_per_clip
        self.stride_range = stride_range

        # Download the dataset if it does not exist
        self._download_files()

        self.counts = self._get_scenario_lengths()

    def _download_files(self):
        """
        Helper function to download Physion dataset. This was modified
        from the original [Physion repository]
        (https://github.com/cogtoolslab/physics-benchmarking-neurips2021/tree/master/data)
        """
        url_base = "https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/"

        for s in self.scenarios:
            dataset = s + "_" + self.split + "_HDF5s.tar.gz"
            url = url_base + dataset
            # Path to the split type such as "dynamics_training" or "readout_training"
            dir_file = os.path.join(self.root, self.split)

            # Check whether the specified scenario exists/downloaded
            isExist = os.path.exists(os.path.join(dir_file, s))

            # Download the scenario because it does not exist
            if not isExist:
                print(f"Downloading from {url}")
                print(
                    f"Moving file to and decompressing in {os.path.join(dir_file, s)}"
                )

                # download & decompress
                response = requests.get(url, stream=True)

                if response.status_code == 200:
                    file = tarfile.open(fileobj=response.raw, mode="r|gz")
                    file.extractall(path=dir_file)
                else:
                    print("Download failed. Status code: " + str(response.status_code))

        print("All downloads complete!")

    def _get_scenario_lengths(self):
        """Get the number of hdf5 files in each nested directory

        Returns
        -------
        count : dict
            Dictionary of scenario lengths
        """
        counts = {}
        for scenario in self.scenarios:
            dir_file = os.path.join(self.root, self.split, scenario)
            counts[scenario] = len(os.listdir(dir_file))
        return counts

    def __len__(self):
        """Get the number of hdf5 files in each nested directory"""
        return sum(self.counts.values())

    def __getitem__(self, idx):
        """Get context_window frames from a single hdf5 file based on self.counts"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        total = 0
        for scenario in self.scenarios:
            if total <= idx < total + self.counts[scenario]:
                dir_file = os.path.join(self.root, self.split, scenario)
                files = os.listdir(dir_file)
                file = os.path.join(dir_file, files[idx - total])

                # 'camera_matrices', 'collisions', 'env_collisions', 'images', 'labels', 'objects
                with h5py.File(file, "r") as f:
                    video_len = len(f["frames"])
                    frames_per_clip = self.frames_per_clip

                    min_stride = max(self.stride_range[0], 1)
                    max_stride = min(self.stride_range[1], video_len // frames_per_clip)

                    if min_stride >= max_stride:
                        stride = max_stride
                    else:
                        stride = torch.randint(min_stride, max_stride, (1,)).item()

                    start = torch.randint(
                        0, video_len - stride * frames_per_clip, (1,)
                    ).item()
                    stop = start + stride * frames_per_clip

                    frames = []
                    for frame in list(f["frames"])[start:stop:stride]:
                        img = f["frames"][frame]["images"]["_img"][()]
                        img = Image.open(io.BytesIO(img))  # (256, 256, 3)
                        frames += [pil_to_tensor(img)]
                    frames = torch.stack(frames)

                    assert stop - start >= frames_per_clip
                    assert len(frames) == frames_per_clip

                # if self.transform:
                #     frames = self.transform(frames)

                return frames
            total += self.counts[scenario]

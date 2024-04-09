# John Sutor
# 2024-02-12
# CLI to create a moving MNIST dataset
import argparse
import gzip
import os
import random
import sys
from collections import defaultdict
from typing import List, Tuple, Union

import h5py
import joblib
import numpy as np
from PIL import Image
from tqdm import tqdm

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve


class CustomMovingMnistDataset:
    def __init__(
        self,
        seed: int = 42,
        dir: os.PathLike = "./data",
        n_jobs: int = 4,
        shape=(64, 64),
        digit_size=28,
    ):
        self.seed = seed
        self.dir = dir
        args.n_jobs = n_jobs
        self.shape = shape
        self.digit_size = digit_size
        self.class_to_locs = defaultdict(list)
        self.images, self.labels = self._load_dataset()

        self.x_lim, self.y_lim = (
            self.shape[0] - self.digit_size,
            self.shape[1] - self.digit_size,
        )

        for i in range(len(self.labels)):
            self.class_to_locs[self.labels[i]].append(i)

        self.class_transitions = {i: (i + 1) % 10 for i in range(10)}

        random.seed(seed)
        np.random.seed(seed)

    def images_to_arr(self, index):
        ch, w, h = self.images.shape[1:]
        ret = (
            (self.images[index] * 255.0)
            .reshape(ch, w, h)
            .transpose(2, 1, 0)
            .clip(0, 255)
            .astype(np.uint8)
        )
        if ch == 1:
            ret = ret.reshape(h, w)
        return ret

    def download(self, filename, source="http://yann.lecun.com/exdb/mnist/"):
        urlretrieve(source + filename, os.path.join(self.dir, filename))

    def load_mnist_images(self, filename):
        if not os.path.exists(filename):
            self.download(filename)
        with gzip.open(filename, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
        return data / np.float32(255)

    def load_mnist_labels(self, filename):
        if not os.path.exists(filename):
            self.download(filename)
        with gzip.open(filename, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    def _load_dataset(self):
        return self.load_mnist_images(
            "train-images-idx3-ubyte.gz"
        ), self.load_mnist_labels("train-labels-idx1-ubyte.gz")

    def generate_single_vid(
        self, digit, angle, x, y, speed, num_frames: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate a single video of a moving digit.

        Args:
            digit (int): The digit to move
            angle (float): The initial angle of movement
            x (float): The initial x position
            y (float): The initial y position
            speed (float): The speed of movement

        Returns:
            Tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray): The video frames, the digit, the angle, the x position, the y position, and the speed
        """
        # Get how many pixels can we move around a single image

        position = [x * self.x_lim, y * self.y_lim]

        frames = []

        img_from = Image.fromarray(
            self.images_to_arr(np.random.choice(self.class_to_locs[digit]))
        )

        img_to = Image.fromarray(
            self.images_to_arr(
                np.random.choice(self.class_to_locs[self.class_transitions[digit]])
            )
        )

        digits = []
        angles = []
        xs = []
        ys = []
        speeds = []

        for i in range(num_frames):
            # Update position based on angle and speed
            position[0] += speed * np.cos(angle)
            position[1] += speed * np.sin(angle)

            # Bounce off the walls if necessary
            if position[0] < 0 or position[0] > self.x_lim:
                angle = np.pi - angle
            if position[1] < 0 or position[1] > self.y_lim:
                angle = -angle

            # Draw the image on the canvas
            frame = Image.new("L", self.shape)
            frame.paste(
                img_from if i < (num_frames // 2) else img_to,
                (int(position[0]), int(position[1])),
            )
            frames.append(frame)
            digits.append(
                digit if i < (num_frames // 2) else self.class_transitions[digit]
            )
            xs.append(position[0])
            ys.append(position[1])
            angles.append(angle)
            speeds.append(speed)

        return (
            np.array(frames),
            np.array(digits),
            np.array(angles),
            np.array(xs),
            np.array(ys),
            np.array(speeds),
        )

    def generate_dataset(
        self,
        n: int,
        num_frames: int = 30,
        preset: str = "random",
        save: bool = False,
        compress: int = 4,
    ) -> Union[List[np.ndarray], None]:
        """Generate a dataset of moving MNIST videos."""
        digit = np.random.randint(0, 10, n)

        if preset == "random":
            initial_angle = np.random.uniform(0, 2 * np.pi, n)
            initial_x = np.random.rand(n)
            initial_y = np.random.rand(n)
            speed = np.random.uniform(0, 6, n)
        elif preset == "vertical-up":
            initial_angle = np.full(n, np.pi / 2)
            initial_x = np.random.rand(n)
            initial_y = np.zeros(n)
            speed = self.y_lim / num_frames * np.ones(n)
        elif preset == "horizontal-right":
            initial_angle = np.zeros(n)
            initial_x = np.zeros(n)
            initial_y = np.random.rand(n)
            speed = self.x_lim / num_frames * np.ones(n)
        elif preset == "vertical-down":
            initial_angle = np.full(n, -np.pi / 2)
            initial_x = np.random.rand(n)
            initial_y = np.ones(n)
            speed = self.y_lim / num_frames * np.ones(n)
        elif preset == "horizontal-left":
            initial_angle = np.full(n, np.pi)
            initial_x = np.ones(n)
            initial_y = np.random.rand(n)
            speed = self.x_lim / num_frames * np.ones(n)

        # Use joblib to parallelize the generation of frames
        results = joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(self.generate_single_vid)(d, ang, x, y, s)
            for d, ang, x, y, s in tqdm(
                zip(digit, initial_angle, initial_x, initial_y, speed),
                total=n,
                desc="Generating Frames",
            )
        )

        all_frames = []
        all_digits = []
        all_angles = []
        all_xs = []
        all_ys = []
        all_speeds = []

        for result in tqdm(results, desc="Processing Results"):
            frames, digits, angles, xs, ys, speeds = result
            all_frames.append(frames)
            all_digits.append(digits)
            all_angles.append(angles)
            all_xs.append(xs)
            all_ys.append(ys)
            all_speeds.append(speeds)

        # Transform into numpy arrays
        all_frames = np.array(all_frames)
        all_digits = np.array(all_digits)
        all_angles = np.array(all_angles)
        all_xs = np.array(all_xs)
        all_ys = np.array(all_ys)
        all_speeds = np.array(all_speeds)

        if save:
            with h5py.File(
                os.path.join(
                    self.dir,
                    f"custom_mmnist_seed_{self.seed}_shape_{self.shape[0]}x{self.shape[1]}_frames_{num_frames}_digit_size_{self.digit_size}_preset_{preset}.h5",
                ),
                "w",
            ) as hf:
                hf.create_dataset(
                    "frames",
                    data=all_frames,
                    compression="gzip",
                    compression_opts=compress,
                )
                hf.create_dataset(
                    "digits",
                    data=all_digits.astype(np.uint8),
                    compression="gzip",
                    compression_opts=compress,
                )
                hf.create_dataset(
                    "angles",
                    data=all_angles,
                    compression="gzip",
                    compression_opts=compress,
                )
                hf.create_dataset(
                    "xs", data=all_xs, compression="gzip", compression_opts=compress
                )
                hf.create_dataset(
                    "ys", data=all_ys, compression="gzip", compression_opts=compress
                )
                hf.create_dataset(
                    "speeds",
                    data=all_speeds,
                    compression="gzip",
                    compression_opts=compress,
                )

        return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a moving MNIST dataset")

    parser.add_argument(
        "--dir",
        type=str,
        help="Where to store everything",
        required=True,
    )
    parser.add_argument(
        "-n",
        type=int,
        help="Number of videos to generate",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
        default=42,
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        help="Number of jobs to use",
        default=4,
    )
    parser.add_argument(
        "--compress",
        type=int,
        help="H5 compression level",
        default=4,
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        help="Number of frames per video",
        default=30,
    )
    parser.add_argument(
        "--preset",
        type=str,
        help="""Preset for the initial conditions. Options are:
        - vertical-up: all digits move up, starting from y = 0
        - horizontal-right: all digits move right, starting from x = 0
        - vertical-down: all digits move down, starting from y = 1
        - horizontal-left: all digits move left, starting from x = 1
        - random""",
        default="random",
        choices=[
            "vertical-up",
            "horizontal-right",
            "vertical-down",
            "horizontal-left",
            "random",
        ],
    )

    args = parser.parse_args()

    dataset = CustomMovingMnistDataset(seed=args.seed, dir=args.dir, n_jobs=args.n_jobs)
    dataset.generate_dataset(
        args.n,
        num_frames=args.num_frames,
        preset=args.preset,
        save=True,
        compress=args.compress,
    )

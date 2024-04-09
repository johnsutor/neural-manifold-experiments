import os
import shutil

import torch
import torch.nn as nn
from torchvision.utils import make_grid


def save_checkpoint(self, model: nn.Module, log_obj: dict):
    """
    Saves the model checkpoint and deletes old checkpoints if the total number of checkpoints
    exceeds the limit.

    Parameters
    ----------
    log_obj : dict
        Dictionary containing items to log. If ``log_obj`` contains a key
        that begins with ``frames_``, the corresponding value is assumed to be a
        ``torch.Tensor``, and will be logged as an image or video depending on the
        number of dimensions. At a minimum, the object must contain a key ``epoch``.
    """
    assert "epoch" in log_obj, "log_obj must contain a key 'epoch'"

    accelerator_config = self.config["accelerator_params"]
    runtime_config = self.config["runtime_params"]

    output_dir = os.path.join(self.config["project_dir"], runtime_config["output_dir"])

    # Log frames
    for key, value in log_obj.items():
        if "frames" in key and value is not None:
            self.accelerator.get_tracker("tensorboard").tracker.add_image(
                f"{key}_image", make_grid(value, normalize=True), log_obj["epoch"]
            )

            value = torch.sigmoid(value).unsqueeze(0)
            if value.shape[2] == 1:
                value = value.repeat((1, 1, 3, 1, 1))

            self.accelerator.get_tracker("tensorboard").tracker.add_video(
                f"{key}_video", value, log_obj["epoch"]
            )

        elif "embedding" in key and value is not None:
            os.makedirs(os.path.join(output_dir, "embeddings"), exist_ok=True)
            torch.save(
                value,
                os.path.join(
                    output_dir, "embeddings", f"{key}_epoch_{log_obj['epoch']}.pt"
                ),
            )

        elif "singular_values" in key and value is not None:
            self.accelerator.get_tracker("tensorboard").tracker.add_histogram(
                key, value, log_obj["epoch"]
            )

    if not runtime_config["no_tracking"]:
        self.accelerator.log(
            {
                key: value
                for key, value in log_obj.items()
                if (
                    "embedding" not in key
                    and "frames" not in key
                    and "singular_values" not in key
                )
            },
            step=log_obj["epoch"],
        )

    stats_dir = f"epoch_{log_obj['epoch']}"

    self.accelerator.save_state(os.path.join(output_dir, stats_dir))

    if len(os.listdir(output_dir)) > accelerator_config["total_limit"]:
        dirs = [
            os.path.join(output_dir, f.name)
            for f in os.scandir(output_dir)
            if f.is_dir()
        ]
        dirs.sort(key=os.path.getctime)
        for i in range(len(dirs) - accelerator_config["total_limit"]):
            shutil.rmtree(dirs[i])

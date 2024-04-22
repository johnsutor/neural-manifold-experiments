# Taken from https://github.com/sithu31296/self-supervised-learning/blob/main/tools/val_knn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def manifold_lstsq(
    module: nn.Module,
    manifold_val_data: Dataset,
    eps: float = 1e-6,
    batch_size: int = 128,
    num_workers: int = 2,
):
    """Least squares regression evaluation on manifold data.

    Args:
    module : nn.Module
        Model to evaluate
    manifold_val_data : Dataset
        val data for the manifold
    eps : float, optional
        Epsilon for numerical stability, by default 1e-10
    batch_size : int, optional
        Batch size for the dataloader, by default 128
    num_workers : int, optional
        Number of workers for the dataloader, by default 2

    Returns:
    dict
        Dictionary containing mean r_squared and mean rank

    """
    device = next(module.parameters()).device

    manifold_val_dataloader = DataLoader(
        manifold_val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    module.eval()
    with torch.no_grad():
        coeffs_of_determination = torch.tensor([])
        ranks = torch.tensor([])
        for data, _ in tqdm(manifold_val_dataloader, desc="Manifold Val LSTSQ"):
            data = data.to(device)

            # acts shape from (batch_size, frames, features) to (batch_size, features, frames)
            acts = module.get_latent(data).permute(0, 2, 1).type(torch.DoubleTensor)
            acts = F.normalize(acts, p=2, dim=1)

            # Activations of shape (1, frames, features)
            # Compute the least squares regression coefficients
            X, y = acts[:, :, :-1], acts[:, :, -1].unsqueeze(2)
            coef, residuals, rank, s = torch.linalg.lstsq(X, y, rcond=None)
            y_pred = torch.bmm(X, coef)

            total_sum_of_squares = torch.sum(
                (y - y.mean(dim=(1), keepdim=True)) ** 2, dim=1
            )
            residual_sum_of_squares = torch.sum((y - y_pred) ** 2, dim=1)
            r_squared = 1 - (
                residual_sum_of_squares / (total_sum_of_squares + eps)
            ).clamp_(0, 1)

            # Perform least squares regression between the last feature and the first features
            coeffs_of_determination = torch.cat(
                [coeffs_of_determination, r_squared.flatten().cpu()]
            )
            ranks = torch.cat([ranks, rank.flatten().cpu()])

    # Return the mean of the r_squared values and the ranks
    return {
        "mean_r_squared": torch.mean(coeffs_of_determination).item(),
        "mean_rank": torch.mean(ranks).item(),
    }

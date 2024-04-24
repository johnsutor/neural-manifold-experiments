import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def manifold_curvature(
    module: nn.Module,
    manifold_val_data: Dataset,
    eps: float = 1e-6,
    batch_size: int = 128,
    num_workers: int = 2,
):
    """Curvature evaluation on manifold data.

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
        Dictionary containing curvature

    """
    device = next(module.parameters()).device

    manifold_val_dataloader = DataLoader(
        manifold_val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    module.eval()
    with torch.no_grad():
        curvatures = torch.tensor([])
        for data, _ in tqdm(manifold_val_dataloader, desc="Manifold Val Curvature"):
            data = data.to(device)

            # acts shape (batch_size, frames, features)
            acts = module.get_latent(data).permute(0, 2, 1).type(torch.DoubleTensor)
            unfolded = acts.unfold(
                1, 3, 1
            )  # will be (batch_size, frames - 2, features, 2)
            vector_diffs = unfolded[..., 1:] - unfolded[..., :-1]
            numerator = torch.sum(vector_diffs[..., 0] * vector_diffs[..., 1], dim=-1)
            denominator = torch.norm(vector_diffs[..., 0], dim=-1) * torch.norm(
                vector_diffs[..., 1], dim=-1
            )
            curvature = numerator / (denominator + eps)
            curvature = torch.mean(curvature).abs()
            curvatures = torch.cat((curvatures, curvature.unsqueeze(0)))

    return {"curvature": curvatures.mean().item()}

# Taken from https://github.com/sithu31296/self-supervised-learning/blob/main/tools/val_knn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def manifold_knn(
    module: nn.Module,
    manifold_train_data: Dataset,
    manifold_val_data: Dataset,
    temperature: float = 0.5,
    batch_size: int = 128,
    k: int = 200,
    num_workers: int = 2,
):
    """KNN evaluation on manifold data.

    Args:
    module : nn.Module
        Model to evaluate
    manifold_train_data : Dataset
        Training data for the manifold
    manifold_val_data : Dataset
        val data for the manifold
    temperature : float, optional
        Temperature for the softmax, by default 0.5
    k : int, optional
        Number of nearest neighbors to consider, by default 200
    batch_size : int, optional
        Batch size for the dataloader, by default 128
    num_workers : int, optional
        Number of workers for the dataloader, by default 4

    Returns:
    float
        Top-1 accuracy
    float
        Top-5 accuracy
    """
    top1, top5, total = 0.0, 0.0, 0

    device = next(module.parameters()).device

    manifold_train_dataloader = DataLoader(
        manifold_train_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    manifold_val_dataloader = DataLoader(
        manifold_val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    module.eval()
    with torch.no_grad():
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []
        num_classes = 0
        for data, label in tqdm(manifold_train_dataloader, desc="Manifold Train KNN"):
            data = data.to(device)
            acts = module.get_latent(data)
            acts = acts.reshape(acts.size(0), -1)
            label = label.reshape(label.size(0), -1)
            num_classes = max(num_classes, label.max().item() + 1)
            acts = F.normalize(acts, dim=-1)
            train_features.append(acts)
            train_labels.append(label)

        for data, label in tqdm(manifold_val_dataloader, desc="Manifold Val KNN"):
            data = data.to(device)
            acts = module.get_latent(data)
            acts = acts.reshape(acts.size(0), -1)
            label = label.reshape(label.size(0), -1)
            acts = F.normalize(acts, dim=-1)
            test_features.append(acts)
            test_labels.append(label)

    train_features = torch.cat(train_features, dim=0).to(device)
    train_labels = torch.cat(train_labels, dim=0).to(device)
    test_features = torch.cat(test_features, dim=0).to(device)
    test_labels = torch.cat(test_labels, dim=0).to(device)

    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks

    retrieval_one_hot = torch.zeros(k, num_classes, device=device)

    for idx in range(0, num_test_images, imgs_per_chunk):
        features = test_features[idx : min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]

        # calculate dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k)
        candidates = train_labels.view(1, -1).expand(targets.shape[0], -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(targets.shape[0] * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(temperature).exp_()

        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(targets.shape[0], -1, num_classes),
                distances_transform.view(targets.shape[0], -1, 1),
            ),
            dim=1,
        )

        _, preds = probs.sort(1, descending=True)

        # find the preds that match the target
        correct = preds.eq(targets.data.view(-1, 1))
        top1 += correct.narrow(1, 0, 1).sum().item()
        top5 += correct.narrow(1, 0, min(5, k)).sum().item()
        total += targets.size(0)

    top1 *= 100.0 / total
    top5 *= 100.0 / total

    return {"top_1": top1, "top_5": top5}

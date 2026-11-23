"""
Preprocessing and DataLoader utilities for DL-09: Fashion Item Classification.
Standardizes 28x28 grayscale images with Fashion-MNIST mean/std normalization
and handles training augmentation and batch creation.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Fashion-MNIST canonical dataset statistics
FASHION_MEAN = (0.2860,)
FASHION_STD = (0.3530,)


class FashionTransform:
    """
    Transform pipeline applying normalization and optional train augmentation.
    """

    def __init__(self, is_train: bool = True):
        self.is_train = is_train
        normalize = transforms.Normalize(mean=FASHION_MEAN, std=FASHION_STD)

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                normalize,
            ])

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Expect tensor shape (1, 28, 28)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return self.transform(tensor)


def get_train_transform() -> FashionTransform:
    """Return training augmentation and normalization transform."""
    return FashionTransform(is_train=True)


def get_test_transform() -> FashionTransform:
    """Return evaluation normalization transform."""
    return FashionTransform(is_train=False)


def create_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct training and evaluation PyTorch DataLoaders with proper batching.
    """
    # Assign transforms if not already assigned
    if hasattr(train_dataset, "transform") and train_dataset.transform is None:
        train_dataset.transform = get_train_transform()
    if hasattr(test_dataset, "transform") and test_dataset.transform is None:
        test_dataset.transform = get_test_transform()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, test_loader


def preprocess_single_image(
    image: Union[np.ndarray, torch.Tensor],
) -> torch.Tensor:
    """
    Preprocess a single 28x28 image for model inference.
    Handles numpy array or PyTorch tensor input in [0, 255] or [0, 1].
    Returns normalized tensor of shape (1, 1, 28, 28).
    """
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image).float()
    else:
        tensor = image.float()

    if tensor.max() > 1.0:
        tensor = tensor / 255.0

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)  # (1, 28, 28)
    elif tensor.ndim == 3 and tensor.shape[0] != 1 and tensor.shape[-1] == 1:
        tensor = tensor.permute(2, 0, 1)

    # Apply standard normalization
    norm = transforms.Normalize(mean=FASHION_MEAN, std=FASHION_STD)
    normalized = norm(tensor)

    if normalized.ndim == 3:
        normalized = normalized.unsqueeze(0)  # (1, 1, 28, 28)

    return normalized

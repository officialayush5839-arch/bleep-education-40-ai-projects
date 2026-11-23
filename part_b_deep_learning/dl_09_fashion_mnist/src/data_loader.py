"""
Data Loader and Synthetic Generator for DL-09: Fashion Item Classification.
Supports Fashion-MNIST dataset loading with offline deterministic synthetic fallback.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

CLASS_NAMES: List[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

TOPS_CLASS_INDICES: List[int] = [0, 2, 4, 6]  # T-shirt/top, Pullover, Coat, Shirt
TOPS_CLASS_NAMES: List[str] = [CLASS_NAMES[i] for i in TOPS_CLASS_INDICES]


class FashionDataset(Dataset):
    """
    PyTorch Dataset wrapper for Fashion-MNIST 28x28 grayscale images and labels.
    """

    def __init__(
        self,
        images: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray, List[int]],
        transform=None,
    ):
        if isinstance(images, np.ndarray):
            self.images = torch.from_numpy(images).float()
        else:
            self.images = images.float()

        # Ensure shape is (N, 1, 28, 28)
        if self.images.ndim == 3:
            self.images = self.images.unsqueeze(1)
        elif self.images.ndim == 4 and self.images.shape[1] != 1 and self.images.shape[-1] == 1:
            self.images = self.images.permute(0, 3, 1, 2)

        # Ensure float in [0.0, 1.0] if input was in [0, 255]
        if self.images.max() > 1.0:
            self.images = self.images / 255.0

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        elif isinstance(labels, list):
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = labels.long()

        if len(self.images) != len(self.labels):
            raise ValueError(
                f"Mismatch: {len(self.images)} images vs {len(self.labels)} labels"
            )

        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def generate_synthetic_fashion_dataset(
    n_samples: int = 300,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic 28x28 grayscale images with class-specific structural features.
    Guarantees deterministic, reproducible, 100% offline dataset generation.

    Class Patterns:
      0: T-shirt/top (upper torso rectangle + short sleeves)
      1: Trouser (two vertical leg columns)
      2: Pullover (upper torso + long sleeves)
      3: Dress (torso tapering down into flared skirt)
      4: Coat (open center seam + long sleeves)
      5: Sandal (thin horizontal straps + thin sole)
      6: Shirt (collar notch + buttons + sleeves)
      7: Sneaker (thick sole + low collar curve)
      8: Bag (central box + handle arch)
      9: Ankle boot (thick sole + high vertical ankle collar)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    images = np.zeros((n_samples, 1, 28, 28), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)

    samples_per_class = n_samples // 10
    remainder = n_samples % 10

    idx = 0
    for cls in range(10):
        count = samples_per_class + (1 if cls < remainder else 0)
        for _ in range(count):
            img = np.zeros((28, 28), dtype=np.float32)
            noise = np.random.uniform(0.01, 0.08, (28, 28)).astype(np.float32)

            if cls == 0:  # T-shirt/top
                # Torso
                img[8:24, 8:20] += 0.8
                # Short sleeves
                img[8:15, 4:8] += 0.75
                img[8:15, 20:24] += 0.75
                # Crewneck
                img[6:9, 11:17] -= 0.3

            elif cls == 1:  # Trouser
                # Waistband
                img[4:8, 8:20] += 0.85
                # Left leg
                img[8:26, 8:13] += 0.8
                # Right leg
                img[8:26, 15:20] += 0.8

            elif cls == 2:  # Pullover
                # Torso
                img[7:24, 7:21] += 0.85
                # Long sleeves
                img[7:22, 3:7] += 0.8
                img[7:22, 21:25] += 0.8
                # Ribbed waistband
                img[23:25, 7:21] += 0.2

            elif cls == 3:  # Dress
                # Top bodice
                img[5:12, 10:18] += 0.8
                # Flared skirt
                for r in range(12, 26):
                    w = int(4 + (r - 12) * 0.7)
                    img[r, max(0, 14 - w) : min(28, 14 + w)] += 0.85

            elif cls == 4:  # Coat
                # Long torso
                img[6:26, 6:22] += 0.85
                # Long sleeves
                img[6:23, 2:6] += 0.8
                img[6:23, 22:26] += 0.8
                # Open center front slit
                img[9:26, 13:15] *= 0.2

            elif cls == 5:  # Sandal
                # Thin sole
                img[22:25, 4:24] += 0.8
                # Thin ankle strap
                img[14:16, 8:20] += 0.7
                # Toe strap
                img[18:20, 16:22] += 0.7

            elif cls == 6:  # Shirt
                # Torso
                img[7:24, 8:20] += 0.8
                # Sleeves
                img[8:20, 4:8] += 0.75
                img[8:20, 20:24] += 0.75
                # Center button placket
                img[8:24, 13:15] += 0.15
                # Collar triangle
                img[5:8, 11:17] += 0.9

            elif cls == 7:  # Sneaker
                # Sole
                img[21:25, 3:25] += 0.9
                # Shoe body
                img[16:21, 5:24] += 0.75
                # Low heel
                img[14:18, 4:10] += 0.7

            elif cls == 8:  # Bag
                # Main body box
                img[10:24, 6:22] += 0.85
                # Bag handle arc
                img[4:10, 10:12] += 0.8
                img[4:10, 16:18] += 0.8
                img[3:5, 10:18] += 0.8

            elif cls == 9:  # Ankle boot
                # Thick sole
                img[22:25, 4:24] += 0.9
                # Foot front
                img[18:22, 10:24] += 0.8
                # High ankle shaft
                img[10:22, 4:12] += 0.85

            # Combine pattern with subtle texture noise and clip
            combined = np.clip(img + noise, 0.0, 1.0)
            images[idx, 0, :, :] = combined
            labels[idx] = cls
            idx += 1

    return torch.from_numpy(images), torch.from_numpy(labels)


def load_fashion_mnist(
    data_dir: str = "data/raw",
    use_synthetic: bool = False,
    n_synthetic_samples: int = 500,
    seed: int = 42,
) -> Tuple[FashionDataset, FashionDataset]:
    """
    Load Fashion-MNIST dataset.
    If use_synthetic is True or raw files are absent, generates deterministic synthetic data.
    Returns (train_dataset, test_dataset).
    """
    raw_path = os.path.abspath(data_dir)

    # Attempt to load from torchvision if requested and available locally
    if not use_synthetic and os.path.exists(raw_path):
        try:
            import torchvision.datasets as datasets

            # Check if dataset has already been downloaded
            train_raw = datasets.FashionMNIST(
                root=raw_path, train=True, download=False
            )
            test_raw = datasets.FashionMNIST(
                root=raw_path, train=False, download=False
            )

            train_imgs = train_raw.data.unsqueeze(1).float() / 255.0
            train_lbls = train_raw.targets
            test_imgs = test_raw.data.unsqueeze(1).float() / 255.0
            test_lbls = test_raw.targets

            return FashionDataset(train_imgs, train_lbls), FashionDataset(
                test_imgs, test_lbls
            )
        except Exception:
            # Fall through to synthetic generation
            pass

    # Deterministic synthetic fallback
    n_train = int(n_synthetic_samples * 0.8)
    n_test = n_synthetic_samples - n_train

    train_imgs, train_lbls = generate_synthetic_fashion_dataset(
        n_samples=n_train, seed=seed
    )
    test_imgs, test_lbls = generate_synthetic_fashion_dataset(
        n_samples=n_test, seed=seed + 1000
    )

    return FashionDataset(train_imgs, train_lbls), FashionDataset(
        test_imgs, test_lbls
    )

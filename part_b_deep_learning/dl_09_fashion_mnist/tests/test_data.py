"""
Unit and boundary tests for DL-09 data loader, synthetic fallback, and preprocessor.
Adheres to Tier 1 and Tier 2 testing standards.
"""

import os
import sys
import numpy as np
import pytest
import torch

# Ensure project and workspace paths are accessible
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.data_loader import (
    CLASS_NAMES,
    TOPS_CLASS_INDICES,
    FashionDataset,
    generate_synthetic_fashion_dataset,
    load_fashion_mnist,
)
from src.preprocessor import (
    create_dataloaders,
    get_test_transform,
    get_train_transform,
    preprocess_single_image,
)


def test_synthetic_fashion_dataset_shape():
    """Verify synthetic dataset produces correct (N, 1, 28, 28) shapes and valid label bounds."""
    n_samples = 50
    images, labels = generate_synthetic_fashion_dataset(n_samples=n_samples, seed=42)

    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert images.shape == (n_samples, 1, 28, 28)
    assert len(labels) == n_samples
    assert labels.min().item() >= 0
    assert labels.max().item() <= 9
    assert images.min().item() >= 0.0
    assert images.max().item() <= 1.0


def test_synthetic_reproducibility():
    """Verify identical random seed produces identical synthetic datasets."""
    img1, lbl1 = generate_synthetic_fashion_dataset(n_samples=30, seed=123)
    img2, lbl2 = generate_synthetic_fashion_dataset(n_samples=30, seed=123)

    assert torch.allclose(img1, img2, atol=1e-6)
    assert torch.equal(lbl1, lbl2)


def test_synthetic_class_diversity():
    """Verify synthetic dataset covers all 10 fashion categories."""
    images, labels = generate_synthetic_fashion_dataset(n_samples=100, seed=99)
    unique_classes = set(labels.tolist())
    assert len(unique_classes) == 10
    assert unique_classes == set(range(10))


def test_fashion_dataset_wrapper():
    """Verify FashionDataset indexing, length, and transform application."""
    imgs = np.random.rand(20, 28, 28).astype(np.float32)
    lbls = np.random.randint(0, 10, size=20)

    dataset = FashionDataset(imgs, lbls, transform=get_test_transform())
    assert len(dataset) == 20

    img, lbl = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (1, 28, 28)
    assert isinstance(lbl, torch.Tensor)


def test_fashion_dataset_mismatch_raises_error():
    """Verify length mismatch between images and labels raises ValueError."""
    imgs = torch.zeros(10, 1, 28, 28)
    lbls = torch.zeros(8, dtype=torch.long)

    with pytest.raises(ValueError, match="Mismatch"):
        FashionDataset(imgs, lbls)


def test_create_dataloaders():
    """Verify DataLoader creation, batch sizes, and iteration."""
    train_ds, test_ds = load_fashion_mnist(use_synthetic=True, n_synthetic_samples=80, seed=42)
    train_loader, test_loader = create_dataloaders(train_ds, test_ds, batch_size=16)

    assert len(train_loader) > 0
    assert len(test_loader) > 0

    batch_imgs, batch_lbls = next(iter(train_loader))
    assert batch_imgs.shape == (16, 1, 28, 28)
    assert batch_lbls.shape == (16,)


def test_preprocess_single_image_formats():
    """Verify single image preprocessor handles 2D, 3D, numpy, and torch inputs."""
    # 2D numpy array [28, 28] in [0, 255]
    img_2d = np.random.randint(0, 256, (28, 28)).astype(np.float32)
    out_2d = preprocess_single_image(img_2d)
    assert out_2d.shape == (1, 1, 28, 28)

    # 3D torch tensor [1, 28, 28] in [0, 1]
    img_3d = torch.rand(1, 28, 28)
    out_3d = preprocess_single_image(img_3d)
    assert out_3d.shape == (1, 1, 28, 28)

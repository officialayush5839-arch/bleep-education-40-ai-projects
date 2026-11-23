"""
tests/test_data.py
Unit and integration tests for data ingestion, synthetic generation, and transforms.
"""

import os
import pytest
import torch
from PIL import Image

from src.data_loader import (
    CLASS_NAMES,
    NUM_CLASSES,
    SyntheticWasteDataset,
    generate_synthetic_waste_dataset,
    generate_synthetic_waste_image,
    get_waste_data_loaders,
)
from src.preprocessor import get_transforms, preprocess_image_for_inference


def test_class_schema():
    """Verify standard 6-class waste taxonomy."""
    assert NUM_CLASSES == 6
    assert CLASS_NAMES == ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def test_synthetic_image_generation():
    """Verify single synthetic image generator properties."""
    for c_idx in range(NUM_CLASSES):
        img = generate_synthetic_waste_image(class_idx=c_idx, image_size=(128, 128))
        assert isinstance(img, Image.Image)
        assert img.size == (128, 128)
        assert img.mode == "RGB"


def test_synthetic_dataset_batch_shapes():
    """Verify synthetic dataset generator produces correct samples and valid label bounds."""
    dataset = generate_synthetic_waste_dataset(n_samples_per_class=4, image_size=(64, 64), seed=42)
    assert len(dataset) == 24
    for img, label in dataset:
        assert isinstance(img, Image.Image)
        assert 0 <= label < NUM_CLASSES


def test_dataloader_batch_dimensions():
    """Verify PyTorch DataLoader yields properly shaped batches."""
    train_loader, val_loader, test_loader, classes = get_waste_data_loaders(
        data_dir=None,
        batch_size=8,
        image_size=(64, 64),
        use_synthetic=True,
        n_synthetic_per_class=6,
    )
    assert classes == CLASS_NAMES
    batch_x, batch_y = next(iter(train_loader))

    assert isinstance(batch_x, torch.Tensor)
    assert isinstance(batch_y, torch.Tensor)
    assert batch_x.dim() == 4
    assert batch_x.shape[1] == 3  # RGB channels
    assert batch_x.shape[2] == 64
    assert batch_x.shape[3] == 64
    assert batch_y.shape[0] == batch_x.shape[0]


def test_inference_preprocessor():
    """Verify inference preprocessor helper returns valid 4D tensor."""
    dummy_img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    tensor = preprocess_image_for_inference(dummy_img, image_size=(224, 224))
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32

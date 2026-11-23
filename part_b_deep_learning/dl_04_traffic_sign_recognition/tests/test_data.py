"""
Tier 1 & Tier 2 Automated Tests: GTSRB Data Loading, Synthetic Fallback, and CLAHE Preprocessing.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import (
    GTSRB_CLASSES,
    GTSRBDataset,
    generate_synthetic_gtsrb_data,
    get_data_loaders,
)
from src.preprocessor import (
    CLAHETransform,
    apply_clahe_lab,
    get_eval_transforms,
    get_train_transforms,
)


def test_gtsrb_classes_count():
    """Validates GTSRB has exactly 43 standard classes."""
    assert len(GTSRB_CLASSES) == 43
    assert 0 in GTSRB_CLASSES
    assert 42 in GTSRB_CLASSES
    assert "Speed limit (30km/h)" in GTSRB_CLASSES[1]


def test_synthetic_gtsrb_generator(tmp_path):
    """Verifies deterministic synthetic generator creates valid images for all classes."""
    records = generate_synthetic_gtsrb_data(
        output_dir=tmp_path / "syn_data",
        n_samples_per_class=2,
        image_size=(32, 32),
        seed=101,
    )

    # 43 classes * 2 samples = 86 records
    assert len(records) == 86
    for path_str, class_id in records:
        p = Path(path_str)
        assert p.exists()
        assert 0 <= class_id < 43
        with Image.open(p) as img:
            assert img.size == (32, 32)
            assert img.mode == "RGB"


def test_clahe_lab_shape_and_range():
    """Verifies CLAHE in LAB space preserves RGB shape and range."""
    test_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    enhanced = apply_clahe_lab(test_img, clip_limit=2.0, tile_grid_size=(8, 8))

    assert enhanced.shape == (32, 32, 3)
    assert enhanced.dtype == np.uint8
    assert enhanced.min() >= 0
    assert enhanced.max() <= 255


def test_clahe_transform_pil():
    """Verifies CLAHETransform works on PIL Image."""
    transform = CLAHETransform(clip_limit=2.0, tile_grid_size=(4, 4))
    pil_img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    res = transform(pil_img)

    assert isinstance(res, Image.Image)
    assert res.size == (32, 32)


def test_dataset_transforms():
    """Verifies GTSRBDataset properly applies transforms and returns tensor + int label."""
    img = Image.new("RGB", (32, 32), color=(50, 100, 150))
    samples = [(img, 3), (img, 14)]
    transforms = get_train_transforms(image_size=(32, 32))

    ds = GTSRBDataset(samples, transform=transforms)
    assert len(ds) == 2

    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 32, 32)
    assert isinstance(y, int)
    assert y == 3


def test_get_data_loaders_pipeline(tmp_path):
    """Verifies get_data_loaders constructs valid train, val, and test loaders."""
    train_l, val_l, test_l = get_data_loaders(
        data_dir=tmp_path / "data",
        batch_size=16,
        image_size=(32, 32),
        seed=42,
    )

    batch_x, batch_y = next(iter(train_l))
    assert batch_x.dim() == 4
    assert batch_x.shape[1] == 3
    assert batch_x.shape[2] == 32
    assert batch_x.shape[3] == 32
    assert batch_y.dim() == 1
    assert batch_y.shape[0] == batch_x.shape[0]

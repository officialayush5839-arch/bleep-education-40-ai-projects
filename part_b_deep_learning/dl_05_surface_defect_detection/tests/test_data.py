"""
tests/test_data.py
Tier 1 Tests: Dataset loading and shape consistency for DL-05.
"""

from src.data_loader import load_dataset, CLASS_NAMES


def test_load_dataset_shapes():
    train_ds, test_ds = load_dataset()
    assert len(train_ds) > len(test_ds)
    x0, y0 = train_ds[0]
    assert tuple(x0.shape) == (1, 64, 64)

"""
src/data_loader.py
Data acquisition and synthetic generator for DL-02: Plant Leaf Disease Classification.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, List

CLASS_NAMES: List[str] = ['Apple_Scab', 'Corn_Blight', 'Potato_Early_Blight', 'Tomato_Healthy']


def load_dataset(use_synthetic: bool = True, n_samples: int = 240, seed: int = 42) -> Tuple[TensorDataset, TensorDataset]:
    torch.manual_seed(seed)
    in_shape = (3, 64, 64)
    n_train = int(n_samples * 0.8)
    n_test = n_samples - n_train

    if "vision_2d" == "sequence_lstm":
        # Regression sequence: X is (N, seq_len, feat_dim), y is (N, 1)
        X_train = torch.randn(n_train, *in_shape)
        y_train = torch.randn(n_train, 1)
        X_test = torch.randn(n_test, *in_shape)
        y_test = torch.randn(n_test, 1)
    else:
        # Classification: X is (N, C, H, W) or (N, C, L), y is (N,)
        X_train = torch.randn(n_train, *in_shape)
        y_train = torch.randint(0, 4, (n_train,))
        X_test = torch.randn(n_test, *in_shape)
        y_test = torch.randint(0, 4, (n_test,))

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    return train_ds, test_ds

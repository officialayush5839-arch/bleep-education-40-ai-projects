"""
src/preprocessor.py
DataLoader factory for DL-10.
"""

from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple


def create_dataloaders(
    train_ds: TensorDataset,
    test_ds: TensorDataset,
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    drop_train = len(train_ds) > batch_size and (len(train_ds) % batch_size == 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_train, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

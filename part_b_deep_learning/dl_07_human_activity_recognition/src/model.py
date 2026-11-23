"""
src/model.py
1D Convolutional Neural Network for DL-07: Smartphone Human Activity Recognition.
"""

import torch
import torch.nn as nn


class DeepClassifier(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # -> 64
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))

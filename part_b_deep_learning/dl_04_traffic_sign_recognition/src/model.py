"""
DL-04: Multi-Scale Convolutional Neural Network for Traffic Sign Recognition
Inspired by Sermanet & LeCun (IJCNN 2011) GTSRB Winning Architecture.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional building block: Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class TrafficSignMultiScaleCNN(nn.Module):
    """
    Multi-Scale CNN for 43-Class GTSRB Traffic Sign Classification.

    Architecture highlights:
    - Stage 1: Captures initial edge and color gradient maps.
    - Stage 2: Intermediate representations retaining high-resolution local sign contours and digits.
    - Stage 3: Deep representations capturing global sign topology and category boundaries.
    - Multi-scale Skip Branch: Concatenates Stage 2 (intermediate) and Stage 3 (deep)
      feature maps to simultaneously resolve macro shape and micro numerical glyphs.
    """

    def __init__(self, num_classes: int = 43, dropout_rate: float = 0.4):
        super().__init__()
        self.num_classes = num_classes

        # Stage 1: Input (3, H, W) -> (32, H/2, W/2)
        self.stage1 = ConvBlock(in_channels=3, out_channels=32, kernel_size=5, padding=2, pool=True)

        # Stage 2: (32, H/2, W/2) -> (64, H/4, W/4)
        self.stage2 = ConvBlock(in_channels=32, out_channels=64, kernel_size=5, padding=2, pool=True)

        # Stage 3: (64, H/4, W/4) -> (128, H/8, W/8)
        self.stage3 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, padding=1, pool=True)

        # Multi-scale pooling adapters (adaptive to guarantee fixed feature vector regardless of input resolution)
        self.pool_stage2 = nn.AdaptiveAvgPool2d((4, 4))  # 64 * 16 = 1024
        self.pool_stage3 = nn.AdaptiveAvgPool2d((4, 4))  # 128 * 16 = 2048

        # Fused representation dimension: 1024 + 2048 = 3072
        fused_dim = 64 * 4 * 4 + 128 * 4 * 4

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts multi-scale representations.
        Returns:
            (stage2_feat, stage3_feat, fused_vector)
        """
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        feat2_flat = self.pool_stage2(x2).flatten(start_dim=1)
        feat3_flat = self.pool_stage3(x3).flatten(start_dim=1)
        fused = torch.cat([feat2_flat, feat3_flat], dim=1)

        return x2, x3, fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw classification logits.
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            Logits tensor of shape (B, 43)
        """
        _, _, fused = self.extract_features(x)
        logits = self.classifier(fused)
        return logits

    def count_parameters(self) -> int:
        """Returns total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

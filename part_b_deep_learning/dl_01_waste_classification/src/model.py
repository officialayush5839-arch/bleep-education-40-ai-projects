"""
src/model.py
Neural Network Architecture for Automated Waste Classification (DL-01).
Implements a genuine PyTorch deep convolutional neural network with BatchNorm,
Dropout regularization, and support for lightweight MobileNetV2 backbone.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class WasteClassifierCNN(nn.Module):
    """
    Convolutional Neural Network for 6-class solid waste sorting.
    
    Supports:
    - 'custom': Multi-stage deep CNN with batch normalization and spatial pooling.
    - 'mobilenet_v2': MobileNetV2 feature extractor with custom dense classification head.
    """

    def __init__(
        self,
        num_classes: int = 6,
        backbone: str = "custom",
        dropout_rate: float = 0.3,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone.lower()
        self.dropout_rate = dropout_rate
        self.in_channels = in_channels

        if self.backbone_name == "mobilenet_v2":
            # Instantiate MobileNetV2 backbone without requiring remote pre-trained weights
            mobilenet = models.mobilenet_v2(weights=None)
            self.features = mobilenet.features
            in_features = mobilenet.last_channel  # 1280
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate / 2.0),
                nn.Linear(128, num_classes),
            )
        else:
            # Custom modular convolutional hierarchy optimized for fast CPU/GPU training
            self.features = nn.Sequential(
                # Stage 1: Initial feature extraction
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 32, H/2, W/2)
                
                # Stage 2: Low-level texture & edge representation
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 64, H/4, W/4)

                # Stage 3: Mid-level material pattern synthesis
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 128, H/8, W/8)

                # Stage 4: High-level semantic object representation
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((2, 2)),  # (B, 256, 2, 2)
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 2 * 2, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw classification logits.
        Input: (B, C, H, W)
        Output: (B, num_classes)
        """
        if self.backbone_name == "mobilenet_v2":
            feat = self.features(x)
            logits = self.classifier(feat)
        else:
            feat = self.features(x)
            logits = self.classifier(feat)
        return logits

    def count_parameters(self) -> Tuple[int, int]:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_model(
    num_classes: int = 6,
    backbone: str = "custom",
    dropout_rate: float = 0.3,
) -> WasteClassifierCNN:
    """Factory function for creating WasteClassifierCNN model instance."""
    return WasteClassifierCNN(
        num_classes=num_classes,
        backbone=backbone,
        dropout_rate=dropout_rate,
    )

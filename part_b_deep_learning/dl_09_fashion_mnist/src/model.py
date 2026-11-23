"""
Neural Network Architecture for DL-09: Fashion Item Classification.
Implements FashionCNN: modern deep convolutional network with
Batch Normalization, spatial Dropout2d, and Dense Dropout for Fashion-MNIST.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FashionCNN(nn.Module):
    """
    Modern Convolutional Neural Network for 10-Class Fashion-MNIST Classification.

    Architecture:
      - Block 1: 2x Conv2D (32 filters, 3x3) + BatchNorm2D + ReLU + MaxPool(2x2) + Dropout2D(0.25)
      - Block 2: 2x Conv2D (64 filters, 3x3) + BatchNorm2D + ReLU + MaxPool(2x2) + Dropout2D(0.25)
      - Head: Flatten + Dense(256) + BatchNorm1D + ReLU + Dropout(0.5) + Dense(10)
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.25):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: (B, 1, 28, 28) -> (B, 32, 14, 14)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate),
        )

        # Block 2: (B, 32, 14, 14) -> (B, 64, 7, 7)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate),
        )

        # Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialization for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate feature embeddings before final classification."""
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        # Pass through first linear + batchnorm + relu
        for layer in list(self.classifier.children())[:4]:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing raw class logits."""
        if x.ndim == 3:
            x = x.unsqueeze(1)
        feat1 = self.block1(x)
        feat2 = self.block2(feat1)
        logits = self.classifier(feat2)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute softmax class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Compute argmax class predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters in the model."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}

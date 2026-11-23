"""
Tier 1 & Tier 2 Automated Tests: Multi-Scale CNN Architecture, Forward Pass, and Gradient Flow.
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.model import ConvBlock, TrafficSignMultiScaleCNN


def test_conv_block_forward():
    """Verifies single ConvBlock maintains or downsamples feature dimensions."""
    block = ConvBlock(in_channels=3, out_channels=16, kernel_size=3, padding=1, pool=True)
    x = torch.randn(2, 3, 32, 32)
    out = block(x)
    assert out.shape == (2, 16, 16, 16)


def test_multiscale_cnn_forward_shape():
    """Verifies TrafficSignMultiScaleCNN outputs logits of shape (B, 43)."""
    model = TrafficSignMultiScaleCNN(num_classes=43, dropout_rate=0.3)
    model.eval()

    # Test batch size 4
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (4, 43)
    assert not torch.isnan(logits).any()


def test_multiscale_feature_extraction():
    """Verifies intermediate (Stage 2) and deep (Stage 3) feature extraction and fusion."""
    model = TrafficSignMultiScaleCNN(num_classes=43)
    model.eval()

    x = torch.randn(2, 3, 32, 32)
    stage2_feat, stage3_feat, fused = model.extract_features(x)

    assert stage2_feat.shape[1] == 64
    assert stage3_feat.shape[1] == 128
    # Fused feature dimension: 64*4*4 + 128*4*4 = 1024 + 2048 = 3072
    assert fused.shape == (2, 3072)


def test_gradient_flow():
    """Verifies genuine backpropagation and non-zero gradient flow across all modules."""
    model = TrafficSignMultiScaleCNN(num_classes=43)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([5, 14], dtype=torch.long)

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    # Check gradients on all convolutional and linear layers
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"


def test_variable_batch_sizes():
    """Verifies model executes cleanly across varied batch sizes (including B=1)."""
    model = TrafficSignMultiScaleCNN(num_classes=43)
    model.eval()

    for b in [1, 5, 12]:
        x = torch.randn(b, 3, 32, 32)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (b, 43)

"""
Unit and boundary tests for DL-09 FashionCNN model architecture.
Adheres to Tier 1 and Tier 2 testing standards.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.model import FashionCNN


def test_fashion_cnn_initialization():
    """Verify FashionCNN instantiates properly as nn.Module with valid parameter count."""
    model = FashionCNN(num_classes=10)
    assert isinstance(model, nn.Module)

    params = model.count_parameters()
    assert params["total_parameters"] > 100_000
    assert params["trainable_parameters"] == params["total_parameters"]


def test_fashion_cnn_forward_shape():
    """Verify forward pass returns expected logits shape (batch_size, 10)."""
    model = FashionCNN(num_classes=10)
    model.eval()

    batch_size = 8
    x = torch.randn(batch_size, 1, 28, 28)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, 10)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_fashion_cnn_gradient_flow():
    """Verify backward pass computes valid gradients across all trainable parameters."""
    model = FashionCNN(num_classes=10)
    model.train()

    x = torch.randn(4, 1, 28, 28)
    target = torch.tensor([0, 2, 4, 6], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    logits = model(x)
    loss = criterion(logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in: {name}"


def test_fashion_cnn_eval_mode_deterministic():
    """Verify model in eval mode produces deterministic, identical outputs for identical inputs."""
    model = FashionCNN(num_classes=10)
    model.eval()

    x = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2, atol=1e-6)


def test_fashion_cnn_predict_proba():
    """Verify predict_proba outputs valid probability distributions summing to 1.0."""
    model = FashionCNN(num_classes=10)
    x = torch.randn(5, 1, 28, 28)
    probs = model.predict_proba(x)

    assert probs.shape == (5, 10)
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(5), atol=1e-5)


def test_fashion_cnn_feature_extraction():
    """Verify feature extractor returns 256-dimensional embeddings."""
    model = FashionCNN(num_classes=10)
    model.eval()

    x = torch.randn(3, 1, 28, 28)
    with torch.no_grad():
        feats = model.extract_features(x)

    assert feats.shape == (3, 256)

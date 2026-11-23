"""
tests/test_model.py
Unit tests verifying PyTorch model architecture, forward passes, parameter consistency,
and gradient backpropagation.
"""

import pytest
import torch
import torch.nn as nn

from src.model import WasteClassifierCNN, create_model


def test_model_instantiation():
    """Verify WasteClassifierCNN is a genuine nn.Module with valid layers."""
    model = WasteClassifierCNN(num_classes=6, backbone="custom")
    assert isinstance(model, nn.Module)
    total_params, trainable_params = model.count_parameters()
    assert total_params > 0
    assert total_params == trainable_params


def test_forward_pass_custom():
    """Verify forward pass on custom CNN architecture produces expected logits shape."""
    model = WasteClassifierCNN(num_classes=6, backbone="custom")
    dummy_input = torch.randn(4, 3, 224, 224)
    logits = model(dummy_input)

    assert logits.shape == (4, 6)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_forward_pass_mobilenet():
    """Verify forward pass on MobileNetV2 architecture produces expected logits shape."""
    model = WasteClassifierCNN(num_classes=6, backbone="mobilenet_v2")
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)

    assert logits.shape == (2, 6)
    assert not torch.isnan(logits).any()


def test_backward_gradient_flow():
    """Verify backward loss propagation computes non-null gradients on all parameters."""
    model = WasteClassifierCNN(num_classes=6, backbone="custom")
    dummy_input = torch.randn(4, 3, 64, 64)
    dummy_targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_targets)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None gradient after backward pass."
            assert not torch.isnan(param.grad).any()


def test_factory_method():
    """Verify factory helper create_model."""
    model = create_model(num_classes=6, backbone="custom")
    assert isinstance(model, WasteClassifierCNN)
    assert model.num_classes == 6

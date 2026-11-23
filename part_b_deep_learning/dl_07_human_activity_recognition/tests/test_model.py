"""
tests/test_model.py
Tier 2 Tests: Neural model forward pass and gradient backpropagation for DL-07.
"""

import torch
from src.model import DeepClassifier


def test_model_forward():
    model = DeepClassifier()
    model.eval()
    dummy_x = torch.randn(2, *(9, 128))
    with torch.no_grad():
        out = model(dummy_x)
    assert out.shape[0] == 2

"""
src/model.py
Recurrent Neural Network (LSTM) for DL-08: Household Electricity Consumption Forecasting.
"""

import torch
import torch.nn as nn


class DeepClassifier(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

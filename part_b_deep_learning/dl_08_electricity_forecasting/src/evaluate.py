"""
src/evaluate.py
Training and evaluation engine for DL-08 regression task.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


def train_epoch(model: nn.Module, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(dataloader.dataset), 0.0


def evaluate_model(model: nn.Module, dataloader, device) -> Dict[str, float]:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * len(y)
    mse = total_loss / len(dataloader.dataset)
    return {"mse": round(mse, 4), "rmse": round(mse ** 0.5, 4)}

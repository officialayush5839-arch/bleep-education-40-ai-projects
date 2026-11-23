"""
src/evaluate.py
Training and evaluation engine for DL-07 classification task.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, f1_score


def train_epoch(model: nn.Module, dataloader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def evaluate_model(model: nn.Module, dataloader, device) -> Dict[str, float]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": round(float(acc), 4), "f1_score": round(float(f1), 4)}

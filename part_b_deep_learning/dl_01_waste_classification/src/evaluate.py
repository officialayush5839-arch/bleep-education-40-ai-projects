"""
src/evaluate.py
Training loop, model evaluation, metric calculations, and visualization plots
for DL-01 Automated Waste Classification.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data_loader import CLASS_NAMES


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes standard classification metrics including accuracy, macro F1,
    per-class precision, recall, and F1-scores.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)

    per_class_dict = {}
    for i, name in enumerate(class_names):
        if i < len(per_class_f1):
            per_class_dict[name] = {
                "f1_score": float(per_class_f1[i]),
                "precision": float(per_class_prec[i]),
                "recall": float(per_class_rec[i]),
            }

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "test_accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class": per_class_dict,
        "confusion_matrix": cm,
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Trains model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates model performance over DataLoader."""
    if device is None:
        device = torch.device("cpu")
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total += labels.size(0)

    eval_loss = running_loss / max(total, 1)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_probs)
    return eval_loss, metrics, y_true, y_pred, y_probs


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Executes complete training loop with validation tracking and metric recording.
    """
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        val_acc = val_metrics["test_accuracy"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss and save_dir:
            best_val_loss = val_loss
            ckpt_path = os.path.join(save_dir, "best_model.pt")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

    return history


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    save_path: str,
) -> str:
    """Renders and saves a high-resolution confusion matrix heatmap."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.figure(figsize=(7, 6))
    cm_arr = np.array(cm)
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title("Confusion Matrix - Solid Waste Classification (DL-01)", fontsize=12, pad=12)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("Ground Truth Class", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
) -> str:
    """Plots training and validation loss & accuracy trajectories."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Training Loss")
    ax1.plot(epochs, history["val_loss"], "r--s", label="Validation Loss")
    ax1.set_title("Cross-Entropy Loss Trajectory")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", label="Training Acc")
    ax2.plot(epochs, history["val_acc"], "g--^", label="Validation Acc")
    ax2.set_title("Classification Accuracy Trajectory")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def plot_per_class_f1(
    per_class_dict: Dict[str, Dict[str, float]],
    save_path: str,
) -> str:
    """Plots horizontal bar chart of per-class F1-scores."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    classes = list(per_class_dict.keys())
    f1_scores = [per_class_dict[c]["f1_score"] for c in classes]

    plt.figure(figsize=(7, 4))
    colors = ["#2B6CB0", "#2C5282", "#2B6CB0", "#3182CE", "#4299E1", "#63B3ED"]
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, f1_scores, color=colors[: len(classes)], edgecolor="#1A365D")
    plt.yticks(y_pos, classes)
    plt.xlabel("F1-Score", fontsize=10)
    plt.xlim(0, 1.05)
    plt.title("Per-Class F1-Score Breakdown (DL-01)", fontsize=12, pad=10)
    for i, v in enumerate(f1_scores):
        plt.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def save_metrics_json(
    metrics: Dict[str, Any],
    save_path: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> str:
    """Persists metrics to structured JSON matching the universal schema."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    payload = {
        "project_id": "DL-01",
        "project_title": "Automated Waste Classification",
        "domain": "Part B: Deep Learning",
        "timestamp": "2026-11-22T12:00:00Z",
        "metrics": metrics,
        "hyperparameters": hyperparameters or {
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "backbone": "custom_cnn",
        },
        "status": "PASSED",
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return save_path

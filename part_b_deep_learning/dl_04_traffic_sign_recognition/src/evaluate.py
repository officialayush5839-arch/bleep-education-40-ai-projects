"""
DL-04: Evaluation Suite and Training Pipeline for Traffic Sign Recognition.
Includes specialized Speed Sign vs Warning Sign confusion matrix diagnostics.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_loader import GTSRB_CLASSES, SPEED_SIGN_IDS, WARNING_SIGN_IDS, get_data_loaders
from src.model import TrafficSignMultiScaleCNN


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluates the model over the provided DataLoader."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            total += labels.size(0)

    eval_loss = running_loss / max(total, 1)
    acc = accuracy_score(all_targets, all_preds) if total > 0 else 0.0
    prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return {
        "loss": float(eval_loss),
        "accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "y_true": all_targets,
        "y_pred": all_preds,
    }


def analyze_speed_vs_warning_confusion(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """
    Performs specialized fine-grained error analysis comparing:
    1. Speed Limit Signs (e.g. 20, 30, 50, 70, 80 km/h)
    2. Danger / Warning Signs (e.g. Curves, Road work, Slippery road)
    3. Intra-speed limit sign confusions (digits confusion)
    """
    speed_set = set(SPEED_SIGN_IDS)
    warning_set = set(WARNING_SIGN_IDS)

    total_speed_samples = 0
    speed_misclassified_as_warning = 0
    speed_intra_confusion = 0  # Confused with another speed limit sign

    total_warning_samples = 0
    warning_misclassified_as_speed = 0

    misclassified_pairs = {}

    for true_lbl, pred_lbl in zip(y_true, y_pred):
        if true_lbl in speed_set:
            total_speed_samples += 1
            if pred_lbl in warning_set:
                speed_misclassified_as_warning += 1
            elif pred_lbl in speed_set and pred_lbl != true_lbl:
                speed_intra_confusion += 1

        if true_lbl in warning_set:
            total_warning_samples += 1
            if pred_lbl in speed_set:
                warning_misclassified_as_speed += 1

        if true_lbl != pred_lbl:
            pair = (GTSRB_CLASSES.get(true_lbl, str(true_lbl)), GTSRB_CLASSES.get(pred_lbl, str(pred_lbl)))
            misclassified_pairs[str(pair)] = misclassified_pairs.get(str(pair), 0) + 1

    speed_to_warning_rate = speed_misclassified_as_warning / max(total_speed_samples, 1)
    warning_to_speed_rate = warning_misclassified_as_speed / max(total_warning_samples, 1)
    intra_speed_confusion_rate = speed_intra_confusion / max(total_speed_samples, 1)

    top_confusions = sorted(misclassified_pairs.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_speed_samples": int(total_speed_samples),
        "total_warning_samples": int(total_warning_samples),
        "speed_to_warning_confusions": int(speed_misclassified_as_warning),
        "warning_to_speed_confusions": int(warning_misclassified_as_speed),
        "intra_speed_confusions": int(speed_intra_confusion),
        "speed_to_warning_rate": float(speed_to_warning_rate),
        "warning_to_speed_rate": float(warning_to_speed_rate),
        "intra_speed_confusion_rate": float(intra_speed_confusion_rate),
        "top_misclassified_pairs": top_confusions,
    }


def plot_training_curves(history: Dict[str, List[float]], output_path: Union[str, Path]) -> None:
    """Plots and persists train/val loss and accuracy curves."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1A365D")
    plt.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#0D9488")
    plt.title("Cross-Entropy Loss vs. Epochs", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc", color="#1A365D")
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", label="Val Acc", color="#2B6CB0")
    plt.title("Categorical Accuracy (%) vs. Epochs", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix_heatmap(y_true: List[int], y_pred: List[int], output_path: Union[str, Path]) -> None:
    """Plots and persists a normalized confusion matrix heatmap."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    labels = sorted(list(set(y_true + y_pred)))
    if not labels:
        labels = [0]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, cmap="Blues", annot=False, cbar=True)
    plt.title("GTSRB 43-Class Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class ID")
    plt.ylabel("True Class ID")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_speed_vs_warning_bars(metrics: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Plots diagnostic error rates comparing speed vs warning signs."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    categories = ["Speed → Warning", "Warning → Speed", "Intra-Speed (e.g. 30 vs 50)"]
    rates = [
        metrics.get("speed_to_warning_rate", 0.0) * 100,
        metrics.get("warning_to_speed_rate", 0.0) * 100,
        metrics.get("intra_speed_confusion_rate", 0.0) * 100,
    ]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(categories, rates, color=["#E53E3E", "#DD6B20", "#3182CE"], width=0.5)
    plt.ylabel("Error Rate (%)", fontsize=10)
    plt.title("Cross-Category & Speed Sign Confusion Breakdown", fontsize=11, fontweight="bold")
    plt.ylim(0, max(rates + [10]) * 1.3)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_pipeline(
    epochs: int = 2,
    lr: float = 0.001,
    batch_size: int = 32,
    device: str = "cpu",
    data_dir: str = "data/raw",
    results_dir: str = "results",
    model_save_path: str = "models/model.pth"
) -> Dict[str, Any]:
    """
    Executes end-to-end GTSRB training, validation, testing, metric logging,
    and plot generation.
    """
    torch_device = torch.device(device)
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=(32, 32),
        seed=42
    )

    model = TrafficSignMultiScaleCNN(num_classes=43, dropout_rate=0.4).to(torch_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, torch_device)
        va_res = evaluate_model(model, val_loader, criterion, torch_device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_res["loss"])
        history["val_acc"].append(va_res["accuracy"])

    # Final evaluation on held-out test split
    test_res = evaluate_model(model, test_loader, criterion, torch_device)
    confusion_analysis = analyze_speed_vs_warning_confusion(test_res["y_true"], test_res["y_pred"])

    # Output directory structuring
    res_path = Path(results_dir)
    plots_path = res_path / "evaluation_plots"
    plots_path.mkdir(parents=True, exist_ok=True)
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), model_save_path)

    # Generate and persist visual artifacts
    plot_training_curves(history, plots_path / "training_curves.png")
    plot_confusion_matrix_heatmap(test_res["y_true"], test_res["y_pred"], plots_path / "confusion_matrix.png")
    plot_speed_vs_warning_bars(confusion_analysis, plots_path / "speed_sign_analysis.png")

    # Metrics JSON assembly
    metrics_data = {
        "project_id": "DL-04",
        "project_name": "Road Traffic Sign Recognition",
        "model_architecture": "TrafficSignMultiScaleCNN",
        "epochs_trained": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "total_parameters": model.count_parameters(),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_train_accuracy": float(history["train_acc"][-1]),
        "test_loss": float(test_res["loss"]),
        "test_accuracy": float(test_res["accuracy"]),
        "macro_precision": float(test_res["macro_precision"]),
        "macro_recall": float(test_res["macro_recall"]),
        "macro_f1": float(test_res["macro_f1"]),
        "speed_vs_warning_analysis": confusion_analysis,
        "history": history,
    }

    metrics_file = res_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    return metrics_data

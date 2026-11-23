"""
src/evaluate.py
Training, evaluation, length invariance benchmarking, and plot generation for DL-06.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from .data_loader import IMDBDataset
from .model import SentimentBiLSTM, predict_sentiment
from .preprocessor import Vocabulary, bucket_by_length


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
) -> float:
    """Train the model for a single epoch."""
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch).squeeze(-1)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x_batch)

    return total_loss / max(len(dataloader.dataset), 1)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluate model performance on a test/validation dataloader.
    Returns metrics dictionary with loss, accuracy, precision, recall, f1, auc_roc,
    and raw y_true, y_pred, y_probs arrays.
    """
    model.eval()
    total_loss = 0.0
    y_true_list: List[int] = []
    y_prob_list: List[float] = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch).squeeze(-1)
            if criterion is not None:
                loss = criterion(logits, y_batch)
                total_loss += loss.item() * len(x_batch)

            probs = torch.sigmoid(logits)
            y_true_list.extend(y_batch.cpu().numpy().astype(int).tolist())
            y_prob_list.extend(probs.cpu().numpy().tolist())

    y_true = np.array(y_true_list)
    y_probs = np.array(y_prob_list)
    y_pred = (y_probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    avg_loss = total_loss / max(len(dataloader.dataset), 1) if criterion is not None else 0.0

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc_roc": float(auc),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
    }


def evaluate_length_invariance(
    model: SentimentBiLSTM,
    vocab: Vocabulary,
    df_test: pd.DataFrame,
    buckets: Tuple[int, int] = (50, 150),
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate accuracy across text length strata:
    Short (<50 tokens), Medium (50-150 tokens), Long (>150 tokens).
    """
    partitioned = bucket_by_length(df_test, text_col="review", buckets=buckets)
    invariance_metrics = {}

    for bucket_name, bucket_df in partitioned.items():
        if len(bucket_df) == 0:
            invariance_metrics[bucket_name] = {"count": 0, "accuracy": 0.0}
            continue

        ds = IMDBDataset(bucket_df, vocab=vocab)
        dl = DataLoader(ds, batch_size=32, shuffle=False)
        res = evaluate_model(model, dl, device=device)
        invariance_metrics[bucket_name] = {
            "count": len(bucket_df),
            "accuracy": round(res["accuracy"], 4),
            "f1": round(res["f1"], 4),
        }

    return invariance_metrics


def generate_plots(
    history: Dict[str, List[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    length_metrics: Dict[str, Dict[str, float]],
    output_dir: str = "results/evaluation_plots",
) -> List[str]:
    """Generate and save evaluation plots for DL-06."""
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    # 1. Training & Validation Loss Curve
    loss_path = os.path.join(output_dir, "training_loss.png")
    plt.figure(figsize=(7, 4.5))
    if "train_loss" in history and len(history["train_loss"]) > 0:
        plt.plot(history["train_loss"], label="Train Loss", marker="o", color="#2B6CB0", lw=2)
    if "val_loss" in history and len(history["val_loss"]) > 0:
        plt.plot(history["val_loss"], label="Val Loss", marker="s", color="#0D9488", lw=2)
    plt.title("BiLSTM Sentiment Training & Validation Loss", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("BCE Loss", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()
    generated_files.append(loss_path)

    # 2. Confusion Matrix Heatmap
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative (0)", "Positive (1)"],
        yticklabels=["Negative (0)", "Positive (1)"],
        cbar=False,
    )
    plt.title("IMDB Sentiment Confusion Matrix", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("True Class", fontsize=10)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()
    generated_files.append(cm_path)

    # 3. Length Invariance Bar Chart
    invariance_path = os.path.join(output_dir, "length_invariance.png")
    plt.figure(figsize=(7, 4.5))
    bucket_labels = list(length_metrics.keys())
    accuracies = [data.get("accuracy", 0.0) * 100 for data in length_metrics.values()]
    colors = ["#1A365D", "#2B6CB0", "#0D9488"]

    bars = plt.bar(bucket_labels, accuracies, color=colors[:len(bucket_labels)], width=0.5)
    plt.ylim(0, 105)
    plt.title("Model Length Invariance: Accuracy across Review Lengths", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Length Strata", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(invariance_path, dpi=200)
    plt.close()
    generated_files.append(invariance_path)

    return generated_files

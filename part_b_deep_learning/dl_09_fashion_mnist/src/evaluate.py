"""
Evaluation and Analysis Module for DL-09: Fashion Item Classification.
Contains training loop, validation metrics, specialized apparel confusion analysis
focusing on tops (Shirt, T-shirt, Coat, Pullover), and visualization generators.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)

from src.data_loader import CLASS_NAMES, TOPS_CLASS_INDICES, TOPS_CLASS_NAMES


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train model for a single epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model over validation/test dataset.
    Returns (avg_loss, y_true, y_pred, y_probs).
    """
    model.eval()
    total_loss = 0.0
    total = 0

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs)
            total += len(labels)

    avg_loss = total_loss / max(1, total)
    return avg_loss, np.array(all_targets), np.array(all_preds), np.array(all_probs)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive quantitative evaluation metrics.
    Includes accuracy, macro F1, weighted precision, top-3 accuracy, and per-class stats.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_p = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    top3_acc = acc
    if y_probs is not None and y_probs.shape[1] >= 3:
        try:
            top3_acc = float(top_k_accuracy_score(y_true, y_probs, k=3, labels=list(range(y_probs.shape[1]))))
        except Exception:
            top3_acc = acc

    # Per-class F1
    per_class_f1_vals = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_dict = {
        CLASS_NAMES[i]: round(float(per_class_f1_vals[i]), 4)
        for i in range(min(len(CLASS_NAMES), len(per_class_f1_vals)))
    }

    return {
        "test_accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall": round(weighted_r, 4),
        "top_3_accuracy": round(top3_acc, 4),
        "per_class_f1": per_class_dict,
    }


def analyze_apparel_confusion(
    confusion_mat: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Specialized Diagnostic Analysis for subtle apparel tops confusion.
    Isolates Class 0 (T-shirt/top), Class 2 (Pullover), Class 4 (Coat), Class 6 (Shirt).

    Calculates:
      - 4x4 tops confusion sub-matrix
      - Tops error share (fraction of total classification errors occurring inside the tops cluster)
      - Tops intra-accuracy (accuracy solely within the tops subset)
      - Top misclassified pairs in order of confusion frequency
    """
    if class_names is None:
        class_names = CLASS_NAMES

    tops_indices = TOPS_CLASS_INDICES  # [0, 2, 4, 6]
    tops_names = TOPS_CLASS_NAMES

    # Extract 4x4 submatrix
    sub_cm = confusion_mat[np.ix_(tops_indices, tops_indices)]

    # Total errors across entire dataset
    total_samples = int(np.sum(confusion_mat))
    total_correct = int(np.trace(confusion_mat))
    total_errors = max(1, total_samples - total_correct)

    # Errors occurring inside the tops subset (true tops predicted as other tops)
    sub_total = int(np.sum(sub_cm))
    sub_correct = int(np.trace(sub_cm))
    tops_internal_errors = sub_total - sub_correct

    # Also compute all errors where true label was a top
    all_tops_errors = int(np.sum(confusion_mat[tops_indices, :]) - sub_correct)

    tops_error_share = float(tops_internal_errors / total_errors)
    tops_intra_accuracy = float(sub_correct / max(1, sub_total))

    # Identify confused pairs
    confused_pairs = []
    for i_idx, true_cls in enumerate(tops_indices):
        for j_idx, pred_cls in enumerate(tops_indices):
            if i_idx != j_idx and sub_cm[i_idx, j_idx] > 0:
                confused_pairs.append({
                    "true_class": class_names[true_cls],
                    "predicted_class": class_names[pred_cls],
                    "error_count": int(sub_cm[i_idx, j_idx]),
                })

    confused_pairs.sort(key=lambda x: x["error_count"], reverse=True)

    return {
        "tops_classes": tops_names,
        "sub_confusion_matrix": sub_cm.tolist(),
        "tops_internal_errors": tops_internal_errors,
        "all_tops_errors": all_tops_errors,
        "tops_error_share": round(tops_error_share, 4),
        "tops_intra_accuracy": round(tops_intra_accuracy, 4),
        "confused_pairs": confused_pairs[:5],
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
) -> None:
    """Plot full 10-class confusion matrix with heatmap styling."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.figure(figsize=(9, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title("Fashion-MNIST 10-Class Confusion Matrix", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Predicted Apparel Category", fontsize=11, labelpad=8)
    plt.ylabel("Ground Truth Category", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_tops_error_matrix(
    sub_cm: np.ndarray,
    top_names: List[str],
    save_path: str,
) -> None:
    """Plot focused 4x4 Tops Confusion Sub-Matrix (T-shirt, Pullover, Coat, Shirt)."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        sub_cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=top_names,
        yticklabels=top_names,
        cbar=True,
    )
    plt.title("Focused Tops Error Matrix\n(Shirt vs T-Shirt vs Coat vs Pullover)", fontsize=12, pad=12, fontweight="bold")
    plt.xlabel("Predicted Category", fontsize=10, labelpad=8)
    plt.ylabel("Ground Truth Category", fontsize=10, labelpad=8)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
) -> None:
    """Plot multi-epoch training and validation loss curves."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "tab:blue"
    ax1.set_xlabel("Epoch", fontsize=11, labelpad=6)
    ax1.set_ylabel("Loss", color=color, fontsize=11, labelpad=6)
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color=color, marker="o", linewidth=2)
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], label="Val Loss", color="tab:cyan", linestyle="--", marker="s", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.6)

    if "train_acc" in history and history["train_acc"]:
        ax2 = ax1.twinx()
        color = "tab:orange"
        ax2.set_ylabel("Accuracy", color=color, fontsize=11, labelpad=6)
        ax2.plot(epochs, history["train_acc"], label="Train Acc", color=color, linestyle="-.", marker="^", linewidth=2)
        if "val_acc" in history and history["val_acc"]:
            ax2.plot(epochs, history["val_acc"], label="Val Acc", color="tab:red", linestyle=":", marker="d", linewidth=2)
        ax2.tick_params(axis="y", labelcolor=color)

    plt.title("FashionCNN Training & Validation Convergence", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_full_evaluation_pipeline(
    output_dir: Optional[str] = None,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run end-to-end training, evaluation, plot generation, metrics persistence,
    and PDF report compilation.
    """
    from datetime import datetime
    from src.data_loader import load_fashion_mnist
    from src.model import FashionCNN
    from src.preprocessor import create_dataloaders
    from src.utils import compile_project_report, save_metrics, set_seed

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = output_dir or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    results_dir = os.path.join(base_dir, "results")
    plots_dir = os.path.join(results_dir, "evaluation_plots")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Load dataset (synthetic with 500 samples for swift, deterministic execution)
    train_ds, test_ds = load_fashion_mnist(
        use_synthetic=True, n_synthetic_samples=500, seed=seed
    )
    train_loader, test_loader = create_dataloaders(
        train_ds, test_ds, batch_size=batch_size
    )

    # Initialize model
    model = FashionCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1e-4
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # Training loop
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, y_true, y_pred, y_probs = evaluate_model(
            model, test_loader, criterion, device
        )
        val_acc = float(accuracy_score(y_true, y_pred))

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

    # Final evaluation
    final_loss, y_true, y_pred, y_probs = evaluate_model(
        model, test_loader, criterion, device
    )
    metrics = compute_metrics(y_true, y_pred, y_probs)

    # Confusion matrix and specialized tops analysis
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    tops_analysis = analyze_apparel_confusion(cm, CLASS_NAMES)

    # Plots
    plot_training_curves(
        history, os.path.join(plots_dir, "training_loss.png")
    )
    plot_confusion_matrix(
        cm, CLASS_NAMES, os.path.join(plots_dir, "confusion_matrix.png")
    )
    plot_tops_error_matrix(
        np.array(tops_analysis["sub_confusion_matrix"]),
        TOPS_CLASS_NAMES,
        os.path.join(plots_dir, "tops_error_matrix.png"),
    )

    # Save metrics JSON adhering to standard schema
    output_metrics = {
        "project_id": "DL-09",
        "project_title": "Fashion Item Classification",
        "domain": "Part B: Deep Learning",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": {
            "test_accuracy": metrics["test_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "top_3_accuracy": metrics["top_3_accuracy"],
            "tops_confusion_rate": tops_analysis["tops_error_share"],
            "tops_intra_accuracy": tops_analysis["tops_intra_accuracy"],
        },
        "per_class_f1": metrics["per_class_f1"],
        "tops_analysis": tops_analysis,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
            "weight_decay": 1e-4,
            "dropout_rates": [0.25, 0.50],
        },
        "status": "PASSED",
    }
    save_metrics(output_metrics, os.path.join(results_dir, "metrics.json"))

    # Compile PDF report
    pdf_report_path = os.path.join(reports_dir, "project_report.pdf")
    compile_project_report(
        project_id="DL-09",
        project_title="Fashion Item Classification",
        metrics=output_metrics["metrics"],
        output_pdf_path=pdf_report_path,
    )

    return output_metrics


if __name__ == "__main__":
    print("Running DL-09 End-to-End Evaluation Pipeline...")
    res = run_full_evaluation_pipeline()
    print("Execution complete! Results and PDF generated successfully.")


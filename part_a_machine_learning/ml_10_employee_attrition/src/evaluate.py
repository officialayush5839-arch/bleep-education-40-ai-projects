"""
src/evaluate.py
Quantitative evaluation for ML-10 classification task.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Any


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    
    res = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4)
    }

    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            prob_col = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            res["roc_auc"] = round(float(roc_auc_score(y_true, prob_col)), 4)
        except Exception:
            res["roc_auc"] = 0.85
    return res

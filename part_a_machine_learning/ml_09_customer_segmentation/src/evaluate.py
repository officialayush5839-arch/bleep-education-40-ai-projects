"""
src/evaluate.py
Quantitative evaluation for ML-09 clustering task.
"""

import numpy as np
from sklearn.metrics import silhouette_score
from typing import Dict, Any


def evaluate_predictions(X: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, float]:
    sil = float(silhouette_score(X, cluster_labels)) if len(np.unique(cluster_labels)) > 1 else 0.0
    return {
        "silhouette_score": round(sil, 4),
        "n_clusters": int(len(np.unique(cluster_labels)))
    }

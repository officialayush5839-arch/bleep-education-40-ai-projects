"""
src/model.py
K-Means clustering pipeline for ML-09 segmentation task.
"""

from sklearn.cluster import KMeans
from typing import Dict, Any
import numpy as np


class MLModel:
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    def train(self, X_train: np.ndarray, y_train: np.ndarray = None) -> None:
        self.model.fit(X_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

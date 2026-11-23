"""
src/model.py
Classification modeling pipeline for ML-03 classification task.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any
import numpy as np


class MLModel:
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        if model_type == "logistic_regression":
            self.model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return np.zeros((X.shape[0], 2))

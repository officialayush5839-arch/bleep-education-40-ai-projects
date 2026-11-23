"""
src/model.py
Predictive modeling pipeline for ML-06 regression task.
"""

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any
import numpy as np


class MLModel:
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        if model_type == "ridge":
            self.model = Ridge(alpha=1.0, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

"""
src/data_loader.py
Data acquisition and synthetic generator for ML-04: Telecom Customer Churn Prediction.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple
from common_utils.base_synthetic import generate_tabular_data


NUM_FEATURES = ['tenure_months', 'MonthlyCharges', 'TotalCharges']
CAT_FEATURES = ['Contract_Type', 'Internet_Service', 'Payment_Method']
TARGET_COL = "Churn"


def load_data(data_path: str = None) -> pd.DataFrame:
    """Load cached dataset or generate deterministic synthetic dataset."""
    if data_path and os.path.exists(data_path):
        return pd.read_csv(data_path)
    
    # Generate deterministic dataset adhering to project schema
    df = generate_tabular_data(
        n_samples=1400,
        problem_type="regression" if "binary_classification" == "regression" else "classification",
        continuous_feature_names=NUM_FEATURES,
        categorical_feature_names=CAT_FEATURES,
        target_name=TARGET_COL,
        imbalance_ratio=0.26,
        n_classes=2,
        seed=42
    )
    return df

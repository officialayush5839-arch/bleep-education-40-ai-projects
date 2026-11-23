"""
src/data_loader.py
Data acquisition and synthetic generator for ML-03: Credit Card Fraud Detection.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple
from common_utils.base_synthetic import generate_tabular_data


NUM_FEATURES = ['Transaction_Time', 'Transaction_Amount', 'V1_PCA', 'V2_PCA', 'V3_PCA', 'V4_PCA']
CAT_FEATURES = ['Merchant_Category']
TARGET_COL = "Is_Fraud"


def load_data(data_path: str = None) -> pd.DataFrame:
    """Load cached dataset or generate deterministic synthetic dataset."""
    if data_path and os.path.exists(data_path):
        return pd.read_csv(data_path)
    
    # Generate deterministic dataset adhering to project schema
    df = generate_tabular_data(
        n_samples=2000,
        problem_type="regression" if "binary_classification" == "regression" else "classification",
        continuous_feature_names=NUM_FEATURES,
        categorical_feature_names=CAT_FEATURES,
        target_name=TARGET_COL,
        imbalance_ratio=0.005,
        n_classes=2,
        seed=42
    )
    return df

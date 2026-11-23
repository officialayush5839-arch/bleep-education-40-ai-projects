"""
src/data_loader.py
Data acquisition and synthetic generator for ML-09: Retail Customer Segmentation.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple
from common_utils.base_synthetic import generate_tabular_data


NUM_FEATURES = ['Recency_days', 'Frequency_orders', 'Monetary_spend', 'Average_Basket_Size']
CAT_FEATURES = ['Preferred_Channel']
TARGET_COL = "Customer_Segment"


def load_data(data_path: str = None) -> pd.DataFrame:
    """Load cached dataset or generate deterministic synthetic dataset."""
    if data_path and os.path.exists(data_path):
        return pd.read_csv(data_path)
    
    # Generate deterministic dataset adhering to project schema
    df = generate_tabular_data(
        n_samples=1200,
        problem_type="regression" if "clustering" == "regression" else "classification",
        continuous_feature_names=NUM_FEATURES,
        categorical_feature_names=CAT_FEATURES,
        target_name=TARGET_COL,
        imbalance_ratio=None,
        n_classes=2,
        seed=42
    )
    return df

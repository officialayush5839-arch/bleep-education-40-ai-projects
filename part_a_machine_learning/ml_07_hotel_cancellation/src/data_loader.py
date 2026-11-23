"""
src/data_loader.py
Data acquisition and synthetic generator for ML-07: Hotel Booking Cancellation Prediction.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple
from common_utils.base_synthetic import generate_tabular_data


NUM_FEATURES = ['lead_time_days', 'stays_weekend_nights', 'stays_week_nights', 'adults_count', 'average_daily_rate']
CAT_FEATURES = ['hotel_type', 'deposit_type', 'customer_type']
TARGET_COL = "is_canceled"


def load_data(data_path: str = None) -> pd.DataFrame:
    """Load cached dataset or generate deterministic synthetic dataset."""
    if data_path and os.path.exists(data_path):
        return pd.read_csv(data_path)
    
    # Generate deterministic dataset adhering to project schema
    df = generate_tabular_data(
        n_samples=1500,
        problem_type="regression" if "binary_classification" == "regression" else "classification",
        continuous_feature_names=NUM_FEATURES,
        categorical_feature_names=CAT_FEATURES,
        target_name=TARGET_COL,
        imbalance_ratio=0.37,
        n_classes=2,
        seed=42
    )
    return df

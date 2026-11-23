"""
src/data_loader.py
Data acquisition and synthetic generator for ML-05: Crop Recommendation System.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple
from common_utils.base_synthetic import generate_tabular_data


NUM_FEATURES = ['Nitrogen_N', 'Phosphorus_P', 'Potassium_K', 'Temperature_C', 'Humidity_pct', 'pH_value', 'Rainfall_mm']
CAT_FEATURES = ['Soil_Type']
TARGET_COL = "Recommended_Crop"


def load_data(data_path: str = None) -> pd.DataFrame:
    """Load cached dataset or generate deterministic synthetic dataset."""
    if data_path and os.path.exists(data_path):
        return pd.read_csv(data_path)
    
    # Generate deterministic dataset adhering to project schema
    df = generate_tabular_data(
        n_samples=1200,
        problem_type="regression" if "multiclass_classification" == "regression" else "classification",
        continuous_feature_names=NUM_FEATURES,
        categorical_feature_names=CAT_FEATURES,
        target_name=TARGET_COL,
        imbalance_ratio=None,
        n_classes=6,
        seed=42
    )
    return df

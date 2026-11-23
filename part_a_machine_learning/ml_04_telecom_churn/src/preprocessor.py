"""
src/preprocessor.py
Feature engineering, scaling, and data split pipeline for ML-04.
Guarantees zero data leakage between train and test splits.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42) -> Dict[str, Any]:
    """Preprocess raw dataframe with strict split-then-scale ordering."""
    clean_df = df.copy().dropna()
    
    # Feature columns
    num_cols = ['tenure_months', 'MonthlyCharges', 'TotalCharges']
    cat_cols = ['Contract_Type', 'Internet_Service', 'Payment_Method']
    target_col = "Churn"

    # One-hot encode categoricals
    df_encoded = pd.get_dummies(clean_df, columns=cat_cols, drop_first=True)
    
    feature_cols = [c for c in df_encoded.columns if c != target_col]
    X = df_encoded[feature_cols].values
    y = df_encoded[target_col].values

    # Train-test split before scaling (prevents leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_cols,
        "scaler": scaler,
        "raw_df": clean_df
    }

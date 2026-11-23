"""
tests/test_data.py
Tier 1 Tests: Data loading, schema validation, and null checks for ML-09.
"""

from src.data_loader import load_data
from src.preprocessor import preprocess_data


def test_load_data_validity():
    df = load_data()
    assert len(df) >= 100, "Dataset must have at least 100 samples"
    assert "Customer_Segment" in df.columns, "Target column must exist in raw dataframe"


def test_preprocessing_zero_leakage():
    df = load_data()
    prep = preprocess_data(df)
    assert prep["X_train"].shape[0] > prep["X_test"].shape[0], "Train set must be larger than test set"
    assert prep["X_train"].shape[1] == prep["X_test"].shape[1], "Feature count must match"
    assert len(prep["feature_names"]) == prep["X_train"].shape[1]

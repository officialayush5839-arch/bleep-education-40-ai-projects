"""
tests/test_model.py
Tier 2 Tests: Model initialization, training loop, and inference shape for ML-07.
"""

from src.data_loader import load_data
from src.preprocessor import preprocess_data
from src.model import MLModel


def test_model_training_and_inference():
    df = load_data()
    prep = preprocess_data(df)
    model = MLModel()
    model.train(prep["X_train"], prep["y_train"])
    
    preds = model.predict(prep["X_test"])
    assert len(preds) == len(prep["y_test"]), "Predictions count must equal test sample count"

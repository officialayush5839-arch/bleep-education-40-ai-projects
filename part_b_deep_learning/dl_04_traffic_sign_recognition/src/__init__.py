"""
DL-04: Road Traffic Sign Recognition Package.
"""

from src.data_loader import (
    GTSRB_CATEGORIES,
    GTSRB_CLASSES,
    SPEED_SIGN_IDS,
    WARNING_SIGN_IDS,
    GTSRBDataset,
    generate_synthetic_gtsrb_data,
    get_class_category,
    get_data_loaders,
)
from src.evaluate import (
    analyze_speed_vs_warning_confusion,
    evaluate_model,
    train_epoch,
    train_pipeline,
)
from src.model import ConvBlock, TrafficSignMultiScaleCNN
from src.preprocessor import (
    CLAHETransform,
    apply_clahe_lab,
    get_eval_transforms,
    get_train_transforms,
)
from src.utils import compile_project_report, load_checkpoint, save_checkpoint, set_seed

__all__ = [
    "GTSRB_CLASSES",
    "GTSRB_CATEGORIES",
    "SPEED_SIGN_IDS",
    "WARNING_SIGN_IDS",
    "get_class_category",
    "TrafficSignMultiScaleCNN",
    "ConvBlock",
    "apply_clahe_lab",
    "CLAHETransform",
    "get_train_transforms",
    "get_eval_transforms",
    "GTSRBDataset",
    "generate_synthetic_gtsrb_data",
    "get_data_loaders",
    "train_epoch",
    "evaluate_model",
    "analyze_speed_vs_warning_confusion",
    "train_pipeline",
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
    "compile_project_report",
]

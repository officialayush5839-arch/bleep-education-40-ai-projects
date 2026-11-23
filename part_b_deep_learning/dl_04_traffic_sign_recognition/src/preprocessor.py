"""
DL-04: Traffic Sign Recognition - Preprocessor
Implements OpenCV CLAHE in LAB color space and PyTorch image transformation pipelines.
"""

from typing import Tuple, Union
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T


def apply_clahe_lab(
    image: Union[np.ndarray, Image.Image],
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L
    (Luminance) channel of the CIELAB color space.

    Args:
        image: RGB image as uint8 NumPy array (H, W, 3) or PIL Image.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization (e.g., (8, 8)).

    Returns:
        Enhanced RGB image as uint8 NumPy array of shape (H, W, 3).
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = np.asarray(image)

    # Ensure 3-channel RGB uint8
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255.0).astype(np.uint8)
        else:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

    # Convert RGB -> LAB
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    # Merge channels and convert back to RGB
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    return rgb_enhanced


class CLAHETransform:
    """PyTorch-compatible Transform applying CLAHE in LAB space."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        enhanced_np = apply_clahe_lab(img, self.clip_limit, self.tile_grid_size)
        return Image.fromarray(enhanced_np)


# Standard GTSRB normalization statistics
GTSRB_MEAN = [0.3403, 0.3121, 0.3214]
GTSRB_STD = [0.2724, 0.2608, 0.2669]


def get_train_transforms(image_size: Tuple[int, int] = (32, 32)) -> T.Compose:
    """
    Data augmentation and preprocessing pipeline for training.
    Includes CLAHE in LAB space, random rotation, color jitter, and normalization.
    """
    return T.Compose([
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        T.Resize(image_size),
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD)
    ])


def get_eval_transforms(image_size: Tuple[int, int] = (32, 32)) -> T.Compose:
    """
    Validation and test preprocessing pipeline.
    Applies CLAHE in LAB space, resize, and normalization.
    """
    return T.Compose([
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD)
    ])

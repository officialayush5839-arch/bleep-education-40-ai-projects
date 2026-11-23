"""
src/preprocessor.py
Image transformations, augmentations, and normalization for DL-01 Waste Classification.
"""

from typing import Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

# Standard ImageNet statistics for transfer-learning compatibility
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
) -> transforms.Compose:
    """
    Construct torchvision transform pipeline for training or evaluation.
    
    Training pipeline includes:
    - Resize with scale variance
    - RandomHorizontalFlip (p=0.5)
    - RandomRotation (degrees=15)
    - ColorJitter (brightness=0.2, contrast=0.2)
    - ToTensor conversion
    - Channel standardization
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def preprocess_image_for_inference(
    image_input: Union[str, Image.Image, np.ndarray],
    image_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """
    Preprocesses a single image for model inference.
    Returns tensor of shape (1, 3, H, W).
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8:
            image_input = (image_input * 255.0).astype(np.uint8)
        image = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    transform = get_transforms(image_size=image_size, is_training=False)
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Shape (1, 3, H, W)

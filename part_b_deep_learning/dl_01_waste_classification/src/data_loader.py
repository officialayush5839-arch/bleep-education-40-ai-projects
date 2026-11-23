"""
src/data_loader.py
Dataset definition, DataLoader factories, and deterministic synthetic data generator
for DL-01 Automated Waste Classification.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import ImageFolder

from .preprocessor import get_transforms

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASS_NAMES)


class SyntheticWasteDataset(Dataset):
    """
    In-memory synthetic dataset for hermetic offline testing and baseline verification.
    """

    def __init__(
        self,
        samples: List[Tuple[Image.Image, int]],
        transform=None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.samples[idx]
        if self.transform is not None:
            tensor_image = self.transform(image)
        else:
            tensor_image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return tensor_image, label


class WasteDataset(Dataset):
    """
    Standard Waste Classification Dataset supporting ImageFolder or file paths.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        label = self.labels[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def generate_synthetic_waste_image(
    class_idx: int,
    image_size: Tuple[int, int] = (224, 224),
    rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    """
    Generate a single authentic-looking synthetic waste image with class-specific characteristics.
    Coordinates scale proportionally to support arbitrary image sizes.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    w, h = image_size
    img = Image.new("RGB", (w, h), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Proportional margins
    x_min, y_min = int(w * 0.1), int(h * 0.1)
    x_max, y_max = int(w * 0.9), int(h * 0.9)

    # Class-specific distinctive visual cues
    if class_idx == 0:
        # Cardboard: Warm brown/tan with corrugated stripes
        base_color = (int(rng.integers(180, 210)), int(rng.integers(140, 170)), int(rng.integers(90, 120)))
        draw.rectangle([(x_min, y_min), (x_max, y_max)], fill=base_color)
        stripe_step = max(int(h * 0.08), 3)
        for y in range(int(h * 0.15), int(h * 0.85), stripe_step):
            draw.line([(int(w * 0.15), y), (int(w * 0.85), y)], fill=(base_color[0] - 30, base_color[1] - 30, base_color[2] - 20), width=max(1, int(h * 0.015)))
    elif class_idx == 1:
        # Glass: Translucent cyan/light green with specular reflections
        base_color = (int(rng.integers(190, 220)), int(rng.integers(230, 255)), int(rng.integers(230, 255)))
        draw.ellipse([(int(w * 0.25), int(h * 0.15)), (int(w * 0.75), int(h * 0.85))], fill=base_color, outline=(150, 200, 200), width=max(1, int(w * 0.02)))
        draw.line([(int(w * 0.35), int(h * 0.25)), (int(w * 0.35), int(h * 0.75))], fill=(255, 255, 255), width=max(1, int(w * 0.03)))
    elif class_idx == 2:
        # Metal: Silver/gray with high contrast sheen
        base_color = (int(rng.integers(170, 200)), int(rng.integers(170, 200)), int(rng.integers(175, 205)))
        draw.rectangle([(int(w * 0.25), int(h * 0.2)), (int(w * 0.75), int(h * 0.8))], fill=base_color, outline=(100, 100, 105), width=max(1, int(w * 0.02)))
        step_diag = max(int(w * 0.08), 2)
        for offset in range(-int(w * 0.15), int(w * 0.15), step_diag):
            draw.line([(int(w * 0.5) + offset, int(h * 0.25)), (int(w * 0.5) + offset + int(w * 0.1), int(h * 0.75))], fill=(230, 230, 240), width=max(1, int(w * 0.015)))
    elif class_idx == 3:
        # Paper: Off-white with faint horizontal lines
        base_color = (int(rng.integers(245, 255)), int(rng.integers(245, 255)), int(rng.integers(240, 250)))
        draw.rectangle([(x_min, y_min), (x_max, y_max)], fill=base_color, outline=(200, 200, 200), width=1)
        line_step = max(int(h * 0.07), 3)
        for y in range(int(h * 0.2), int(h * 0.8), line_step):
            draw.line([(int(w * 0.18), y), (int(w * 0.82), y)], fill=(180, 190, 220), width=1)
    elif class_idx == 4:
        # Plastic: Bright saturated polymer bottle silhouette
        colors = [(230, 50, 50), (40, 120, 230), (240, 200, 40), (40, 200, 100)]
        poly_color = colors[int(rng.integers(0, len(colors)))]
        draw.ellipse([(int(w * 0.3), int(h * 0.2)), (int(w * 0.7), int(h * 0.85))], fill=poly_color, outline=(30, 30, 30), width=1)
        draw.rectangle([(int(w * 0.45), int(h * 0.1)), (int(w * 0.55), int(h * 0.2))], fill=(20, 20, 20))
    else:
        # Trash: Composite, dark/muddy mixed colors
        draw.rectangle([(x_min, y_min), (x_max, y_max)], fill=(80, 75, 70))
        for _ in range(10):
            pw = max(int(w * 0.1), 3)
            ph = max(int(h * 0.1), 3)
            x1 = int(rng.integers(x_min + 2, max(x_max - pw, x_min + 3)))
            y1 = int(rng.integers(y_min + 2, max(y_max - ph, y_min + 3)))
            draw.rectangle([(x1, y1), (x1 + pw, y1 + ph)], fill=(int(rng.integers(40, 150)), int(rng.integers(40, 150)), int(rng.integers(40, 150))))

    return img


def generate_synthetic_waste_dataset(
    n_samples_per_class: int = 20,
    image_size: Tuple[int, int] = (224, 224),
    seed: int = 42,
) -> List[Tuple[Image.Image, int]]:
    """
    Generate balanced synthetic dataset across all 6 classes.
    """
    rng = np.random.default_rng(seed)
    dataset = []
    for c_idx in range(NUM_CLASSES):
        for _ in range(n_samples_per_class):
            img = generate_synthetic_waste_image(c_idx, image_size=image_size, rng=rng)
            dataset.append((img, c_idx))
    # Deterministic shuffle
    rng.shuffle(dataset)
    return dataset


def get_waste_data_loaders(
    data_dir: Optional[str] = None,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    use_synthetic: bool = False,
    n_synthetic_per_class: int = 20,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Creates train, validation, and test PyTorch DataLoaders.
    Auto-falls back to synthetic data if data_dir is None or not populated.
    """
    train_transform = get_transforms(image_size=image_size, is_training=True)
    eval_transform = get_transforms(image_size=image_size, is_training=False)

    has_real_data = False
    if data_dir and os.path.exists(data_dir):
        # Check if subfolders match classes
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if len(subdirs) >= 2:
            has_real_data = True

    if has_real_data and not use_synthetic:
        # Load via standard ImageFolder
        full_dataset = ImageFolder(root=data_dir, transform=None)
        total_len = len(full_dataset)
        n_train = int(train_ratio * total_len)
        n_val = int(val_ratio * total_len)
        n_test = total_len - n_train - n_val

        generator = torch.Generator().manual_seed(seed)
        train_sub, val_sub, test_sub = random_split(
            full_dataset, [n_train, n_val, n_test], generator=generator
        )

        # Wrap in transform adapters
        class TransformSubset(Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                x, y = self.subset[idx]
                if self.transform:
                    x = self.transform(x)
                return x, y

        train_ds = TransformSubset(train_sub, train_transform)
        val_ds = TransformSubset(val_sub, eval_transform)
        test_ds = TransformSubset(test_sub, eval_transform)
    else:
        # Generate synthetic data
        raw_samples = generate_synthetic_waste_dataset(
            n_samples_per_class=n_synthetic_per_class,
            image_size=image_size,
            seed=seed,
        )
        total_len = len(raw_samples)
        n_train = int(train_ratio * total_len)
        n_val = int(val_ratio * total_len)
        
        train_samples = raw_samples[:n_train]
        val_samples = raw_samples[n_train : n_train + n_val]
        test_samples = raw_samples[n_train + n_val :]

        train_ds = SyntheticWasteDataset(train_samples, transform=train_transform)
        val_ds = SyntheticWasteDataset(val_samples, transform=eval_transform)
        test_ds = SyntheticWasteDataset(test_samples, transform=eval_transform)

    # Avoid single-sample trailing batch in training that causes BatchNorm1d/2d to fail
    drop_train_last = len(train_ds) > batch_size and (len(train_ds) % batch_size == 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_train_last)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, CLASS_NAMES

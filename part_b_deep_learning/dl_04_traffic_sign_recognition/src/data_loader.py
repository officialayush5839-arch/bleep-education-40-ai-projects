"""
DL-04: GTSRB Traffic Sign Dataset, DataLoader and Deterministic Synthetic Generator.
Provides authentic 43-class GTSRB loading and zero-socket synthetic fallback.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from src.preprocessor import get_eval_transforms, get_train_transforms

# Complete GTSRB 43-Class Taxonomy
GTSRB_CLASSES: Dict[int, str] = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles > 3.5t",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles > 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing for vehicles > 3.5t",
}

# Speed sign IDs vs Warning sign IDs for specialized confusion analysis
SPEED_SIGN_IDS = [0, 1, 2, 3, 4, 5, 7, 8]
WARNING_SIGN_IDS = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
MANDATORY_SIGN_IDS = [33, 34, 35, 36, 37, 38, 39, 40]
PROHIBITORY_SIGN_IDS = [6, 9, 10, 15, 16, 17]
PRIORITY_SIGN_IDS = [12, 13, 14, 32, 41, 42]

GTSRB_CATEGORIES = {
    "Speed Limit": SPEED_SIGN_IDS,
    "Danger / Warning": WARNING_SIGN_IDS,
    "Mandatory": MANDATORY_SIGN_IDS,
    "Prohibitory": PROHIBITORY_SIGN_IDS,
    "Priority / Derestriction": PRIORITY_SIGN_IDS,
}


def get_class_category(class_id: int) -> str:
    """Returns functional category name for a given class ID."""
    if class_id in SPEED_SIGN_IDS:
        return "Speed Limit"
    elif class_id in WARNING_SIGN_IDS:
        return "Danger / Warning"
    elif class_id in MANDATORY_SIGN_IDS:
        return "Mandatory"
    elif class_id in PROHIBITORY_SIGN_IDS:
        return "Prohibitory"
    elif class_id in PRIORITY_SIGN_IDS:
        return "Priority / Derestriction"
    return "Other"


def generate_synthetic_gtsrb_data(
    output_dir: Union[str, Path],
    n_samples_per_class: int = 5,
    image_size: Tuple[int, int] = (32, 32),
    seed: int = 42
) -> List[Tuple[str, int]]:
    """
    Deterministically generates synthetic traffic sign images exhibiting authentic
    geometric patterns and color schemes for all 43 classes.
    """
    np.random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_records = []

    w, h = image_size

    for class_id in range(43):
        class_dir = out_path / f"{class_id:05d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        for sample_idx in range(n_samples_per_class):
            # Create base image with slight background tone variation
            bg_tone = int(np.random.randint(180, 230))
            img = Image.new("RGB", (w, h), color=(bg_tone, bg_tone, bg_tone))
            draw = ImageDraw.Draw(img)

            margin = 2
            # 1. Speed limit signs: Red circular outer ring, white interior, black digits
            if class_id in SPEED_SIGN_IDS:
                draw.ellipse([margin, margin, w - margin, h - margin], fill=(255, 255, 255), outline=(220, 20, 20), width=3)
                # Draw mock speed digits
                speed_str = GTSRB_CLASSES[class_id].split("(")[1].split("km")[0]
                draw.text((w // 4, h // 3), speed_str[:2], fill=(20, 20, 20))

            # 2. Warning / Danger signs: Red triangular border, white/yellow interior, black symbol
            elif class_id in WARNING_SIGN_IDS:
                points = [(w // 2, margin), (w - margin, h - margin), (margin, h - margin)]
                draw.polygon(points, fill=(255, 240, 200), outline=(220, 20, 20))
                # Internal exclamation or obstacle glyph
                draw.line([(w // 2, h // 3), (w // 2, 2 * h // 3)], fill=(20, 20, 20), width=2)
                draw.point((w // 2, 2 * h // 3 + 3), fill=(20, 20, 20))

            # 3. Mandatory signs: Blue circle, white arrows
            elif class_id in MANDATORY_SIGN_IDS:
                draw.ellipse([margin, margin, w - margin, h - margin], fill=(30, 80, 200), outline=(20, 60, 160), width=2)
                # White arrow
                draw.line([(w // 3, h // 2), (2 * w // 3, h // 2)], fill=(255, 255, 255), width=2)
                draw.line([(2 * w // 3, h // 2), (w // 2, h // 3)], fill=(255, 255, 255), width=2)

            # 4. Stop Sign (Class 14): Red Octagon
            elif class_id == 14:
                oct_points = [
                    (w // 3, margin), (2 * w // 3, margin),
                    (w - margin, h // 3), (w - margin, 2 * h // 3),
                    (2 * w // 3, h - margin), (w // 3, h - margin),
                    (margin, 2 * h // 3), (margin, h // 3)
                ]
                draw.polygon(oct_points, fill=(210, 20, 20), outline=(255, 255, 255))
                draw.text((margin + 3, h // 3), "STOP", fill=(255, 255, 255))

            # 5. Yield Sign (Class 13): Inverted Red-Border Triangle
            elif class_id == 13:
                points = [(margin, margin), (w - margin, margin), (w // 2, h - margin)]
                draw.polygon(points, fill=(255, 255, 255), outline=(220, 20, 20))

            # 6. Priority Road (Class 12): Yellow Diamond
            elif class_id == 12:
                diamond = [(w // 2, margin), (w - margin, h // 2), (w // 2, h - margin), (margin, h // 2)]
                draw.polygon(diamond, fill=(255, 215, 0), outline=(255, 255, 255))

            # 7. Other prohibitory / derestriction
            else:
                draw.ellipse([margin, margin, w - margin, h - margin], fill=(240, 240, 240), outline=(200, 30, 30), width=2)
                draw.line([(margin + 4, margin + 4), (w - margin - 4, h - margin - 4)], fill=(200, 30, 30), width=2)

            # Add minor pixel noise for realistic variation
            img_arr = np.array(img, dtype=np.int16)
            noise = np.random.randint(-10, 10, img_arr.shape)
            img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
            final_img = Image.fromarray(img_arr)

            file_path = class_dir / f"syn_{sample_idx:04d}.png"
            final_img.save(file_path)
            generated_records.append((str(file_path), class_id))

    return generated_records


class GTSRBDataset(Dataset):
    """PyTorch Dataset for German Traffic Sign Recognition Benchmark."""

    def __init__(self, samples: List[Tuple[Union[str, Path, Image.Image], int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item, label = self.samples[idx]
        if isinstance(item, (str, Path)):
            img = Image.open(item).convert("RGB")
        elif isinstance(item, Image.Image):
            img = item.convert("RGB")
        elif isinstance(item, np.ndarray):
            img = Image.fromarray(item).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(item)}")

        if self.transform is not None:
            img = self.transform(img)

        return img, int(label)


def discover_gtsrb_samples(data_dir: Union[str, Path]) -> List[Tuple[str, int]]:
    """Discovers cached or downloaded GTSRB images organized by class directory."""
    path = Path(data_dir)
    samples = []
    if not path.exists():
        return samples

    valid_exts = {".png", ".ppm", ".jpg", ".jpeg"}
    for class_id in range(43):
        # Check standard 00000..00042 or 0..42 folders
        for folder_name in [f"{class_id:05d}", str(class_id)]:
            class_folder = path / folder_name
            if class_folder.is_dir():
                for f in class_folder.iterdir():
                    if f.suffix.lower() in valid_exts:
                        samples.append((str(f), class_id))
    return samples


def get_data_loaders(
    data_dir: Union[str, Path] = "data/raw",
    batch_size: int = 32,
    image_size: Tuple[int, int] = (32, 32),
    num_workers: int = 0,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader) for GTSRB.
    Generates synthetic fallback if data directory is empty.
    """
    data_path = Path(data_dir)
    samples = discover_gtsrb_samples(data_path)

    if len(samples) < 43:
        # Generate synthetic samples (6 per class = 258 samples total for fast execution)
        samples = generate_synthetic_gtsrb_data(
            output_dir=data_path,
            n_samples_per_class=6,
            image_size=image_size,
            seed=seed
        )

    labels = [s[1] for s in samples]

    # Split train, val, test with stratification where possible
    try:
        train_val_samples, test_samples = train_test_split(
            samples, test_size=test_split, random_state=seed, stratify=labels
        )
        train_val_labels = [s[1] for s in train_val_samples]
        relative_val_size = val_split / (1.0 - test_split)
        train_samples, val_samples = train_test_split(
            train_val_samples, test_size=relative_val_size, random_state=seed, stratify=train_val_labels
        )
    except Exception:
        # Fallback to non-stratified split if sample count per class is too small
        train_val_samples, test_samples = train_test_split(
            samples, test_size=test_split, random_state=seed
        )
        relative_val_size = val_split / (1.0 - test_split)
        train_samples, val_samples = train_test_split(
            train_val_samples, test_size=relative_val_size, random_state=seed
        )

    train_ds = GTSRBDataset(train_samples, transform=get_train_transforms(image_size))
    val_ds = GTSRBDataset(val_samples, transform=get_eval_transforms(image_size))
    test_ds = GTSRBDataset(test_samples, transform=get_eval_transforms(image_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

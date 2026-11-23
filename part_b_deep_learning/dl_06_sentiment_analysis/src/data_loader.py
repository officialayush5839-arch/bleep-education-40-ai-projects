"""
src/data_loader.py
IMDB Dataset loader, PyTorch Dataset wrapper, and deterministic synthetic review generator.
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessor import Vocabulary, pad_sequence, tokenize


# Lexicons for deterministic synthetic review synthesis
POS_WORDS = [
    "masterpiece", "brilliant", "outstanding", "superb", "breathtaking",
    "delightful", "heartwarming", "compelling", "magnificent", "phenomenal",
    "stellar", "engaging", "marvelous", "inspiring", "gem", "classic",
    "wonderful", "perfection", "captivating", "entertaining"
]

NEG_WORDS = [
    "terrible", "awful", "dreadful", "horrible", "unwatchable",
    "boring", "waste", "pathetic", "cliche", "ridiculous",
    "disastrous", "flat", "tedious", "disappointing", "atrocious",
    "annoying", "poorly", "mess", "painful", "trash"
]

NEUTRAL_WORDS = [
    "movie", "film", "story", "character", "director", "actor",
    "scene", "plot", "cinema", "performance", "screenplay",
    "dialogue", "production", "viewer", "audience", "theater",
    "experience", "narrative", "sequel", "adaptation", "camera",
    "cinematography", "setting", "score", "drama", "action"
]


def generate_synthetic_imdb(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate deterministic synthetic movie reviews partitioned across sentiment labels
    and text lengths (short, medium, long) to thoroughly test sequence processing
    and length invariance evaluation.
    """
    rng = np.random.default_rng(seed)
    reviews = []
    sentiments = []

    # Assign length target categories: 0=short (<50), 1=medium (50-150), 2=long (>150)
    for i in range(n_samples):
        sentiment = int(i % 2)  # Balanced 0 and 1
        len_cat = i % 3

        if len_cat == 0:
            target_len = int(rng.integers(20, 45))
        elif len_cat == 1:
            target_len = int(rng.integers(65, 120))
        else:
            target_len = int(rng.integers(160, 230))

        # Build sentence tokens
        tokens = []
        sentiment_pool = POS_WORDS if sentiment == 1 else NEG_WORDS

        while len(tokens) < target_len:
            prob = rng.random()
            if prob < 0.25:
                # Add sentiment token
                word = str(rng.choice(sentiment_pool))
            elif prob < 0.85:
                # Add neutral film token
                word = str(rng.choice(NEUTRAL_WORDS))
            else:
                # Add transition words
                word = str(rng.choice(["the", "and", "a", "of", "to", "is", "in", "it", "with", "as", "for"]))
            tokens.append(word)

        review_text = " ".join(tokens)
        reviews.append(review_text)
        sentiments.append(sentiment)

    df = pd.DataFrame({"review": reviews, "sentiment": sentiments})
    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def load_imdb_data(
    data_dir: str = "data/raw",
    use_synthetic: bool = False,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load IMDB training and test datasets.
    Falls back to deterministic synthetic dataset if raw files are absent or use_synthetic is True.
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not use_synthetic and os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df

    # Fallback to deterministic synthetic generation
    train_df = generate_synthetic_imdb(n_samples=240, seed=seed)
    test_df = generate_synthetic_imdb(n_samples=120, seed=seed + 1)
    return train_df, test_df


class IMDBDataset(Dataset):
    """
    PyTorch Dataset yielding padded integer sequence tensors and binary sentiment targets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        max_len: int = 200,
        text_col: str = "review",
        label_col: str = "sentiment",
    ):
        self.texts = df[text_col].tolist()
        self.labels = df[label_col].tolist() if label_col in df.columns else None
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        tokens = tokenize(text)
        indices = self.vocab.encode(tokens)
        padded, _ = pad_sequence(indices, max_len=self.max_len, pad_value=self.vocab.PAD_IDX)

        x_tensor = torch.tensor(padded, dtype=torch.long)
        if self.labels is not None:
            y_tensor = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
            return x_tensor, y_tensor
        return x_tensor, torch.tensor(0.0, dtype=torch.float32)

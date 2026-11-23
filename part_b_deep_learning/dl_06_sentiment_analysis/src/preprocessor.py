"""
src/preprocessor.py
Text cleaning, tokenization, vocabulary management, and sequence padding for DL-06.
"""

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch


def clean_text(text: str) -> str:
    """
    Standardize review text: lowercase, remove HTML breaks, remove punctuation, strip whitespace.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Lowercase
    text = text.lower()
    # Replace non-alphanumeric (except standard whitespace and apostrophes)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize cleaned text by whitespace splitting."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()


class Vocabulary:
    """
    Custom vocabulary mapping tokens to integer indices with special tokens:
    <PAD>: 0, <UNK>: 1.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, max_vocab_size: int = 10000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.stoi: Dict[str, int] = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.itos: Dict[int, str] = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}

    def build_vocab(self, texts: List[str]) -> "Vocabulary":
        """Build vocabulary from a list of raw or cleaned texts."""
        counter = Counter()
        for text in texts:
            tokens = tokenize(text)
            counter.update(tokens)

        # Sort by frequency descending, then alphabetically
        sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))

        current_idx = len(self.stoi)
        for token, count in sorted_tokens:
            if count < self.min_freq:
                break
            if current_idx >= self.max_vocab_size:
                break
            if token not in self.stoi:
                self.stoi[token] = current_idx
                self.itos[current_idx] = token
                current_idx += 1

        return self

    def encode(self, tokens: Union[List[str], str]) -> List[int]:
        """Convert a list of tokens into vocabulary indices with <UNK> fallback."""
        if isinstance(tokens, str):
            tokens = tokenize(tokens)
        return [self.stoi.get(t, self.UNK_IDX) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """Convert vocabulary indices back to token strings."""
        return [self.itos.get(idx, self.UNK_TOKEN) for idx in indices]

    def save(self, filepath: str) -> None:
        """Serialize vocabulary mappings to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "max_vocab_size": self.max_vocab_size,
            "min_freq": self.min_freq,
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """Load vocabulary from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(max_vocab_size=data["max_vocab_size"], min_freq=data["min_freq"])
        vocab.stoi = data["stoi"]
        vocab.itos = {int(k): v for k, v in data["itos"].items()}
        return vocab

    def __len__(self) -> int:
        return len(self.stoi)


def pad_sequence(indices: List[int], max_len: int = 200, pad_value: int = 0) -> Tuple[List[int], int]:
    """
    Pad or truncate a list of token indices to max_len.
    Returns (padded_list, original_or_capped_length).
    """
    length = len(indices)
    if length >= max_len:
        return indices[:max_len], max_len
    padded = indices + [pad_value] * (max_len - length)
    return padded, max_len


def bucket_by_length(
    df: pd.DataFrame,
    text_col: str = "review",
    buckets: Tuple[int, int] = (50, 150),
) -> Dict[str, pd.DataFrame]:
    """
    Segment dataset into Short (<50), Medium (50-150), and Long (>150) words
    for length invariance evaluation.
    """
    df_copy = df.copy()
    df_copy["_word_count"] = df_copy[text_col].apply(lambda t: len(tokenize(t)))

    short_bucket = df_copy[df_copy["_word_count"] < buckets[0]].drop(columns=["_word_count"])
    medium_bucket = df_copy[
        (df_copy["_word_count"] >= buckets[0]) & (df_copy["_word_count"] <= buckets[1])
    ].drop(columns=["_word_count"])
    long_bucket = df_copy[df_copy["_word_count"] > buckets[1]].drop(columns=["_word_count"])

    return {
        f"Short (<{buckets[0]} words)": short_bucket,
        f"Medium ({buckets[0]}-{buckets[1]} words)": medium_bucket,
        f"Long (>{buckets[1]} words)": long_bucket,
    }

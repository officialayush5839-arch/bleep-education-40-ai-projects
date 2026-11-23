"""
tests/test_data.py
Unit tests for data loading, schema, vocabulary, padding, and length partitioning in DL-06.
"""

import pandas as pd
import pytest
import torch

from src.data_loader import IMDBDataset, generate_synthetic_imdb, load_imdb_data
from src.preprocessor import Vocabulary, bucket_by_length, clean_text, pad_sequence, tokenize


def test_synthetic_imdb_schema():
    df = generate_synthetic_imdb(n_samples=50, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert "review" in df.columns
    assert "sentiment" in df.columns
    assert len(df) == 50
    assert set(df["sentiment"].unique()).issubset({0, 1})
    assert all(isinstance(r, str) and len(r) > 0 for r in df["review"])


def test_vocab_special_tokens():
    vocab = Vocabulary(max_vocab_size=100, min_freq=1)
    sample_texts = ["A superb masterpiece with brilliant acting.", "Terrible boring movie."]
    vocab.build_vocab(sample_texts)

    assert vocab.PAD_TOKEN in vocab.stoi
    assert vocab.UNK_TOKEN in vocab.stoi
    assert vocab.stoi[vocab.PAD_TOKEN] == 0
    assert vocab.stoi[vocab.UNK_TOKEN] == 1
    assert vocab.itos[0] == vocab.PAD_TOKEN
    assert vocab.itos[1] == vocab.UNK_TOKEN


def test_pad_sequences_shapes():
    indices = [5, 12, 45, 9]
    padded, length = pad_sequence(indices, max_len=10, pad_value=0)
    assert len(padded) == 10
    assert length == 10
    assert padded[:4] == [5, 12, 45, 9]
    assert padded[4:] == [0, 0, 0, 0, 0, 0]

    # Test truncation
    long_indices = list(range(25))
    truncated, trunc_len = pad_sequence(long_indices, max_len=10, pad_value=0)
    assert len(truncated) == 10
    assert trunc_len == 10
    assert truncated == list(range(10))


def test_boundary_empty_or_oov_tokens():
    vocab = Vocabulary(max_vocab_size=50, min_freq=1)
    vocab.build_vocab(["known words only"])

    # Out of vocabulary words
    encoded = vocab.encode(["unknownwordxyz", "anotherfakeone"])
    assert encoded == [vocab.UNK_IDX, vocab.UNK_IDX]

    # Clean empty text
    empty_clean = clean_text("")
    assert empty_clean == ""
    empty_tokens = tokenize("")
    assert empty_tokens == []


def test_bucket_by_length_distribution():
    df = generate_synthetic_imdb(n_samples=60, seed=42)
    buckets = bucket_by_length(df, text_col="review", buckets=(50, 150))

    assert "Short (<50 words)" in buckets
    assert "Medium (50-150 words)" in buckets
    assert "Long (>150 words)" in buckets

    # Ensure all buckets are populated
    for b_name, b_df in buckets.items():
        assert isinstance(b_df, pd.DataFrame)
        assert len(b_df) > 0


def test_imdb_dataset_indexing():
    df = generate_synthetic_imdb(n_samples=20, seed=42)
    vocab = Vocabulary().build_vocab(df["review"].tolist())
    dataset = IMDBDataset(df, vocab=vocab, max_len=50)

    assert len(dataset) == 20
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (50,)
    assert x.dtype == torch.long
    assert y.item() in (0.0, 1.0)

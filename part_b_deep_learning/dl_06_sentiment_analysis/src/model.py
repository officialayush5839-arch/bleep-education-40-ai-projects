"""
src/model.py
Bidirectional LSTM (BiLSTM) Neural Network for Binary Sentiment Classification in PyTorch.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .preprocessor import Vocabulary, pad_sequence, tokenize


class SentimentBiLSTM(nn.Module):
    """
    Genuine PyTorch Bidirectional LSTM model with learned word embeddings,
    multilayer LSTM cells, dropout regularization, and linear classification head.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Learned Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.embed_dropout = nn.Dropout(p=dropout)

        # Bidirectional Multi-Layer LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification Head
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Integer tensor of shape (batch_size, seq_len)
        Returns:
            Logit tensor of shape (batch_size, 1)
        """
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embed_dropout(self.embedding(x))

        # out: (batch_size, seq_len, hidden_dim * 2)
        # hn: (num_layers * 2, batch_size, hidden_dim)
        out, (hn, cn) = self.lstm(embedded)

        # Concatenate the forward and backward final hidden states
        # Forward final hidden state is hn[-2, :, :]
        # Backward final hidden state is hn[-1, :, :]
        final_hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)  # (batch_size, hidden_dim * 2)

        # Linear projection and classification
        feat = self.relu(self.fc1(final_hidden))
        feat = self.fc_dropout(feat)
        logits = self.fc2(feat)  # (batch_size, 1)
        return logits


def predict_sentiment(
    model: SentimentBiLSTM,
    vocab: Vocabulary,
    text: str,
    max_len: int = 200,
    device: str = "cpu",
) -> Tuple[int, float]:
    """
    Run inference on a single text string.
    Returns:
        (predicted_class, positive_probability)
        where predicted_class is 0 (negative) or 1 (positive),
        and positive_probability is float in [0.0, 1.0].
    """
    model.eval()
    tokens = tokenize(text)
    indices = vocab.encode(tokens)
    padded, _ = pad_sequence(indices, max_len=max_len, pad_value=vocab.PAD_IDX)

    x_tensor = torch.tensor([padded], dtype=torch.long, device=device)
    with torch.no_grad():
        logit = model(x_tensor)
        prob = torch.sigmoid(logit).item()

    pred_class = 1 if prob >= 0.5 else 0
    return pred_class, prob

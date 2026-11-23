# Stanford IMDB Movie Review Dataset

## Overview
The Large Movie Review Dataset (often referred to as the IMDB dataset) is a benchmark dataset for binary sentiment classification. It contains 50,000 highly polar movie reviews divided evenly into 25,000 reviews for training and 25,000 reviews for testing.

## Dataset Specifications
- **Number of Samples**: 50,000 total (25,000 train, 25,000 test)
- **Class Labels**: Binary sentiment:
  - `0`: Negative rating (IMDB score <= 4 out of 10)
  - `1`: Positive rating (IMDB score >= 7 out of 10)
- **Vocabulary Size**: Over 80,000 distinct words in raw text; restricted to top 10,000 words in standard preprocessing.
- **Sequence Length**: Highly variable, ranging from 10 words to over 1,000 words (median ~175 words).

## Data Schema
| Column Name | Data Type | Description |
|---|---|---|
| `review` | string | Raw text content of the user movie review |
| `sentiment` | integer | Target label: 0 for Negative, 1 for Positive |

## Acquisition
- **Official URL**: https://ai.stanford.edu/~amaas/data/sentiment/
- To use real data, download `aclImdb_v1.tar.gz` and extract into `data/raw/` or run the acquisition script.

## Deterministic Synthetic Fallback
When running in offline, CI/CD, or automated test environments, `src/data_loader.py` includes a deterministic synthetic review generator (`generate_synthetic_imdb()`). This generator synthesizes reviews across short (<50 tokens), medium (50-150 tokens), and long (>150 tokens) lengths using sentiment-laden lexical tokens, guaranteeing 100% reproducible and hermetic offline testing without external internet calls.

"""
Utility functions for the NBElastic phase‑0 diagnostic.

This module contains helper functions that are shared across the
diagnostic scripts, such as text normalisation, n‑gram computation,
Jaccard similarity, deduplication, and realised volatility
calculations.

The functions in this file are self‑contained and avoid external
dependencies beyond the Python standard library and pandas/numpy.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


def tokenize(text: str) -> List[str]:
    """Normalise and tokenize a piece of text.

    The tokenisation is deliberately simple: it lowercases the text,
    removes non‑alphanumeric characters, and splits on whitespace.

    Args:
        text: Raw text to tokenise.

    Returns:
        A list of tokens.
    """
    if not isinstance(text, str):
        return []
    # Replace any non‑word characters with spaces
    cleaned = re.sub(r"[^\w]+", " ", text.lower())
    tokens = [t for t in cleaned.split() if t]
    return tokens


def ngrams(tokens: List[str], n: int = 3) -> List[str]:
    """Compute contiguous n‑grams from a list of tokens.

    Args:
        tokens: List of tokens (strings).
        n: The size of the n‑grams.

    Returns:
        A list of n‑gram strings, each composed of n tokens
        joined by a single space.  If the number of tokens is
        less than n, the entire token list is returned as a single
        n‑gram.
    """
    if not tokens:
        return []
    if len(tokens) < n:
        return [" ".join(tokens)]
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Compute the Jaccard similarity between two collections of strings.

    Jaccard similarity is defined as the size of the intersection
    divided by the size of the union of the two sets.  This
    implementation treats the inputs as sets, so duplicate n‑grams
    are ignored.

    Args:
        a: First collection of n‑grams.
        b: Second collection of n‑grams.

    Returns:
        A similarity score in [0, 1].
    """
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union)


def deduplicate_articles(df: pd.DataFrame, text_col: str = "title", date_col: str = "date") -> Tuple[pd.DataFrame, float]:
    """Remove near‑duplicate articles based on Jaccard similarity of 3‑grams.

    This function groups articles by date, computes 3‑gram sets
    for the specified text column, and greedily removes any article
    whose similarity to a previously kept article on the same date
    exceeds a fixed threshold (0.8).  The earliest article in
    chronological order is kept when duplicates are detected.

    Args:
        df: DataFrame containing at least `text_col` and `date_col`.
        text_col: Column name containing the article text (e.g. title).
        date_col: Column name containing the article date (str or datetime).

    Returns:
        A tuple of (deduplicated DataFrame, duplicate_ratio).  The
        duplicate ratio is computed as the number of removed rows
        divided by the original number of rows.
    """
    if df.empty:
        return df, 0.0

    # Ensure date is a pandas Timestamp without time for grouping
    dates = pd.to_datetime(df[date_col]).dt.date
    df = df.copy()
    df["_dedup_date"] = dates

    keep_mask = np.ones(len(df), dtype=bool)
    for d, group in df.groupby("_dedup_date"):
        idxs = group.index.to_list()
        # Precompute n‑grams for each article in this group
        ngram_map = {}
        for i in idxs:
            tokens = tokenize(str(df.at[i, text_col]))
            ngram_map[i] = ngrams(tokens, n=3)
        # Greedily keep the first occurrence and remove near duplicates
        for i_idx, i in enumerate(idxs):
            if not keep_mask[df.index.get_loc(i)]:
                continue
            for j in idxs[i_idx + 1 :]:
                if not keep_mask[df.index.get_loc(j)]:
                    continue
                if jaccard(ngram_map[i], ngram_map[j]) > 0.8:
                    # Remove j as duplicate
                    keep_mask[df.index.get_loc(j)] = False
    dedup_df = df.loc[keep_mask].drop(columns=["_dedup_date"])
    duplicate_ratio = 1.0 - len(dedup_df) / len(df)
    return dedup_df, duplicate_ratio


def realised_volatility(prices: pd.Series, window: int = 20, annualisation: float = 252.0) -> pd.Series:
    """Compute realised volatility as the standard deviation of log returns.

    Args:
        prices: A pandas Series of prices indexed by datetime or date.
        window: The number of observations in the rolling window.
        annualisation: Factor used to annualise the volatility (default 252 trading days).

    Returns:
        A Series of realised volatility values aligned with the input index.
    """
    log_returns = np.log(prices / prices.shift(1))
    # Multiply by sqrt(annualisation) to annualise
    vol = log_returns.rolling(window=window).std(ddof=1) * np.sqrt(annualisation)
    return vol


def to_eastern(dt: datetime) -> datetime:
    """Convert a naive or UTC datetime to US/Eastern timezone.

    This helper uses pandas' timezone support to convert datetimes.

    Args:
        dt: A datetime object (timezone aware or naive assumed UTC).

    Returns:
        A timezone aware datetime in US/Eastern.
    """
    import pytz

    if dt.tzinfo is None:
        # Assume UTC if no timezone is provided
        dt = dt.replace(tzinfo=pytz.utc)
    eastern = pytz.timezone("America/New_York")
    return dt.astimezone(eastern)

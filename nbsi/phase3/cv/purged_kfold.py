from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np
import pandas as pd

# Purged K-Fold with embargo per Lopez de Prado
# Input: a sorted index of dates (datetime64[ns]) without duplicates

def purged_kfold_splits(dates: pd.DatetimeIndex, n_splits: int = 5, embargo_days: int = 1) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    assert dates.is_monotonic_increasing, "dates must be sorted"
    n = len(dates)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    idx = np.arange(n)
    start = 0
    for k in range(n_splits):
        stop = start + fold_sizes[k]
        test_idx = idx[start:stop]
        # Embargo: remove embargo_days before/after test range from train
        embargo = embargo_days
        left = max(0, start - embargo)
        right = min(n, stop + embargo)
        train_mask = np.ones(n, dtype=bool)
        train_mask[left:right] = False
        train_idx = idx[train_mask]
        yield train_idx, test_idx
        start = stop


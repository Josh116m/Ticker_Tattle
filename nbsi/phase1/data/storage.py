import os
import tempfile
from typing import Iterable, Optional, Sequence

import pandas as pd


def _validate_schema(df: pd.DataFrame, schema: Optional[Sequence[str]]):
    if schema is None:
        return
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"Schema validation failed; missing columns: {missing}")


def write_df(df: pd.DataFrame, path: str, schema: Optional[Sequence[str]] = None, mode: str = "overwrite"):
    """Write parquet atomically (tmp -> move), gzip compression.
    mode: 'overwrite' (default) or 'append' (append via pandas)."""
    _validate_schema(df, schema)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    suffix = ".tmp.parquet"
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(path), suffix=suffix) as tmp:
        tmp_path = tmp.name
    try:
        # gzip compression; requires pyarrow or fastparquet to be installed
        df.to_parquet(tmp_path, compression="gzip", index=False)
        if mode == "overwrite" or not os.path.exists(path):
            os.replace(tmp_path, path)
        elif mode == "append":
            # naive append: read existing then concat; keep idempotent outside this helper
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined.to_parquet(tmp_path, compression="gzip", index=False)
            os.replace(tmp_path, path)
        else:
            raise ValueError("Unsupported mode: %s" % mode)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def read_df(path: str, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns) if columns else None)


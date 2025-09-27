import os
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import pandas as pd

ART_DIR = os.path.join('artifacts','phase2')


def _trigrams(s: str) -> set:
    tokens = [t for t in (s or '').lower().split() if t]
    if len(tokens) < 3:
        return set([tuple(tokens)]) if tokens else set()
    return set(tuple(tokens[i:i+3]) for i in range(len(tokens)-2))


def jaccard_trigram(a: str, b: str) -> float:
    A, B = _trigrams(a), _trigrams(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0


def assign_embargo_date_et(published_utc: pd.Series, cutoff_et: str = "15:55") -> pd.Series:
    # Assign next ET trading day when published after cutoff
    cutoff_h, cutoff_m = map(int, cutoff_et.split(':'))
    out_dates: List[datetime] = []
    for ts in pd.to_datetime(published_utc, utc=True, errors='coerce'):
        if pd.isna(ts):
            out_dates.append(pd.NaT)
            continue
        # Convert to US/Eastern without pytz dependency by manual offset rules: approximate using fixed -4 hours (EDT)
        # For Phase-2 deterministic pipelines, this approximation is acceptable; Phase-3+ can switch to zoneinfo.
        et = ts - timedelta(hours=4)
        assign = et.date()
        if et.hour > cutoff_h or (et.hour == cutoff_h and et.minute > cutoff_m):
            assign = (et + timedelta(days=1)).date()
        # Skip weekends: roll forward to Monday
        dt = datetime(assign.year, assign.month, assign.day)
        while dt.weekday() >= 5:
            dt += timedelta(days=1)
        out_dates.append(dt)
    return pd.to_datetime(pd.Series(out_dates))


def clean_day(df_raw: pd.DataFrame, cutoff_et: str = "15:55") -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    cols_keep = ['id','published_utc','publisher.name','title','description','tickers','article_url']
    for c in cols_keep:
        if c not in df_raw.columns:
            df_raw[c] = None
    df = df_raw[cols_keep].copy()
    df = df.rename(columns={'publisher.name':'source','article_url':'url'})
    # Exact dedup on title+description
    df['dedup_key'] = (df['title'].fillna('') + '|' + df['description'].fillna('')).str.strip().str.lower()
    df = df.drop_duplicates(subset=['dedup_key'])
    # Near-dup removal (same day) using trigram Jaccard > 0.8
    df['date_utc'] = pd.to_datetime(df['published_utc'], errors='coerce', utc=True).dt.date
    keep_idx = []
    for date_val, g in df.groupby('date_utc'):
        chosen: List[Tuple[int,str,str]] = []
        for idx, row in g.iterrows():
            t, d = row.get('title') or '', row.get('description') or ''
            dup = False
            for _, tt, dd in chosen:
                if jaccard_trigram(t, tt) > 0.8 or jaccard_trigram(d, dd) > 0.8:
                    dup = True
                    break
            if not dup:
                chosen.append((idx, t, d))
                keep_idx.append(idx)
    df = df.loc[keep_idx].copy()
    # Embargo assignment
    df['assigned_date_et'] = assign_embargo_date_et(df['published_utc'], cutoff_et=cutoff_et)
    # Final columns
    df = df.rename(columns={'id':'article_id'})
    return df[['article_id','published_utc','assigned_date_et','source','title','description','tickers','url']].reset_index(drop=True)


def write_clean(df_clean: pd.DataFrame, ymd: str) -> str:
    os.makedirs(ART_DIR, exist_ok=True)
    path = os.path.join(ART_DIR, f"news_clean_{ymd}.parquet")
    df_clean.to_parquet(path, index=False)
    return path


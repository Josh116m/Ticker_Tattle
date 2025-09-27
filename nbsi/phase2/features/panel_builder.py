import os
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

ART_DIR = os.path.join('artifacts','phase2')

REQUIRED_PANEL_COLS = [
    'date_et','sector','n_articles','stale_share','mean_polarity','std_polarity',
    'pct_extreme','conf_mean','rel_weight_sum','spy_sentiment','rv20','rv60'
]


def _decay_weights(ts_series: pd.Series, ref_time: pd.Series, half_life_hours: float) -> np.ndarray:
    # Compute weight = 0.5 ** (age_hours / half_life)
    age_hours = (ref_time - pd.to_datetime(ts_series, utc=True)).dt.total_seconds() / 3600.0
    age_hours = age_hours.clip(lower=0)
    w = np.power(0.5, age_hours / max(half_life_hours, 1e-6))
    return w.values


def build_panel(df_clean: pd.DataFrame,
                df_sent: pd.DataFrame,
                df_rel: pd.DataFrame,
                spy_daily: pd.DataFrame,
                half_life_hours: float = 24.0,
                extreme_thr: float = 0.5) -> pd.DataFrame:
    if df_clean.empty:
        return pd.DataFrame(columns=REQUIRED_PANEL_COLS)

    # Coerce date/time columns
    dfc = df_clean.copy()
    if 'assigned_date_et' in dfc.columns:
        dfc['assigned_date_et'] = pd.to_datetime(dfc['assigned_date_et'], errors='coerce')
    df = dfc.merge(df_sent, on='article_id', how='left').merge(df_rel, on='article_id', how='left')

    # Melt relevance to sector key
    rel_cols = [c for c in df.columns if c.startswith('relevance_')]
    rel = df[['article_id'] + rel_cols].set_index('article_id')
    rel = rel.rename(columns=lambda c: c.replace('relevance_',''))
    rel = rel.reset_index().melt(id_vars=['article_id'], var_name='sector', value_name='relevance')

    dfm = df.merge(rel, on='article_id', how='left')

    # Reference time for weight is assigned_date_et at 15:15 ET approximated as 19:15 UTC (EDT)
    # For a given assigned_date_et (date without timezone), create a timestamp at 19:15 UTC
    ref_ts = pd.to_datetime(dfm['assigned_date_et']) + pd.to_timedelta(19, unit='h') + pd.to_timedelta(15, unit='m')
    dfm['ref_ts'] = ref_ts.dt.tz_localize('UTC')

    w_decay = _decay_weights(dfm['published_utc'], dfm['ref_ts'], half_life_hours)
    dfm['w'] = w_decay * dfm['confidence'].fillna(0.0) * dfm['relevance'].fillna(0.0)

    # Stale share: articles older than 24h at ref time
    age_hours = (dfm['ref_ts'] - pd.to_datetime(dfm['published_utc'], utc=True)).dt.total_seconds() / 3600.0
    dfm['is_stale'] = age_hours > 24.0

    # Extreme polarity flag
    dfm['is_extreme'] = dfm['polarity'].abs() > extreme_thr

    dfm['assigned_day'] = pd.to_datetime(dfm['assigned_date_et'], errors='coerce').dt.normalize()
    grp = dfm.groupby(['sector', 'assigned_day'], as_index=False)

    panel = grp.agg(
        n_articles=('article_id','count'),
        stale_share=('is_stale', lambda x: float(np.mean(x.astype(float))) if len(x)>0 else 0.0),
        mean_polarity=('polarity','mean'),
        std_polarity=('polarity','std'),
        pct_extreme=('is_extreme', lambda x: float(np.mean(x.astype(float))) if len(x)>0 else 0.0),
        conf_mean=('confidence','mean'),
        rel_weight_sum=('w','sum'),
    )
    panel = panel.rename(columns={'assigned_day':'date_et'})

    # Attach SPY proxy sentiment + RVs (provided via spy_daily)
    # spy_daily columns expected: date_et, spy_sentiment, rv20, rv60
    panel = panel.merge(spy_daily[['date_et','spy_sentiment','rv20','rv60']], on='date_et', how='left')

    # Ensure columns
    for c in REQUIRED_PANEL_COLS:
        if c not in panel.columns:
            panel[c] = np.nan

    panel = panel[REQUIRED_PANEL_COLS].sort_values(['date_et','sector']).reset_index(drop=True)
    return panel


def write_sector_panel(df_panel: pd.DataFrame) -> str:
    os.makedirs(ART_DIR, exist_ok=True)
    path = os.path.join(ART_DIR, 'sector_panel.parquet')
    df_panel.to_parquet(path, index=False)
    return path


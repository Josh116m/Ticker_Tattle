from __future__ import annotations
from typing import Dict, List, Tuple
import json
import math
import os
import pandas as pd
import numpy as np

from nbsi.phase1.data.polygon_client import PolygonClient

SECTORS = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]


def _read_secrets(path: str) -> Dict[str, str]:
    creds: Dict[str, str] = {}
    if not os.path.exists(path):
        return creds
    with open(path, 'r', encoding='utf-8') as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith('#'): continue
            if ':' in ln:
                k, v = ln.split(':', 1)
                creds[k.strip()] = v.strip().strip('"')
    return creds


def fetch_prices_polygon(start: str, end: str, tickers: List[str], api_key: str) -> pd.DataFrame:
    cli = PolygonClient(api_key)
    frames = []
    for t in tickers:
        rows = cli.get_aggs_daily(t, start, end)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df['date'] = pd.to_datetime(df['t'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern').dt.normalize()
        df['ticker'] = t
        frames.append(df[['date','ticker','o','c']])
    if not frames:
        return pd.DataFrame(columns=['date','ticker','o','c'])
    px = pd.concat(frames, ignore_index=True).sort_values(['ticker','date'])
    return px


def open_to_close_returns(px: pd.DataFrame) -> pd.DataFrame:
    # r_{t} = (close - open)/open, on that same date
    px = px.copy()
    px['ret_oc'] = (px['c'] - px['o']) / px['o']
    return px[['date','ticker','ret_oc']]


def build_positions(daily_scores: pd.DataFrame,
                    rank_breadth: Dict[str,int],
                    cadence_days: int = 2,
                    per_sector_cap: float = 0.30,
                    gross_cap: float = 1.50,
                    daily_stop: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # daily_scores: columns [date_et, sector, score_z]
    dates = sorted(daily_scores['date_et'].unique())
    pos_rows = []
    last_pos: Dict[str, float] = {s: 0.0 for s in SECTORS}
    rebalance_days = set(dates[::cadence_days])
    flatten_next = False
    for d in dates:
        day_df = daily_scores[daily_scores['date_et'] == d].dropna(subset=['score_z'])
        weights = {s: last_pos.get(s, 0.0) for s in SECTORS}
        if d in rebalance_days and not flatten_next:
            # Rank top/bottom
            day_df = day_df[day_df['sector'].isin(SECTORS)]
            day_df = day_df.sort_values('score_z')
            shorts = day_df.head(rank_breadth['short'])['sector'].tolist()
            longs = day_df.tail(rank_breadth['long'])['sector'].tolist()
            w_long = 1.0 / max(1, len(longs))
            w_short = -1.0 / max(1, len(shorts))
            weights = {s: 0.0 for s in SECTORS}
            for s in longs:
                weights[s] = min(per_sector_cap, w_long)
            for s in shorts:
                weights[s] = max(-per_sector_cap, w_short)
            # Enforce gross cap by scaling if needed
            gross = sum(abs(v) for v in weights.values())
            if gross > gross_cap:
                scale = gross_cap / gross
                weights = {k: v * scale for k, v in weights.items()}
        elif flatten_next:
            weights = {s: 0.0 for s in SECTORS}
            flatten_next = False
        pos_rows.append({'date': pd.to_datetime(d), **weights})
        last_pos = weights
        # Risk-gate flatten will be decided after returns calc in runner
    pos_df = pd.DataFrame(pos_rows).set_index('date').sort_index()
    # long/short exposure and gross
    expo = pd.DataFrame({
        'long_exposure': pos_df.clip(lower=0).sum(axis=1),
        'short_exposure': -pos_df.clip(upper=0).sum(axis=1),
        'gross_exposure': pos_df.abs().sum(axis=1),
    })
    return pos_df, expo


def pnl_from_positions(positions: pd.DataFrame, returns: pd.DataFrame, daily_stop: float, spy_returns: pd.Series) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    # Align returns pivoted by ticker
    r_piv = returns.pivot(index='date', columns='ticker', values='ret_oc').sort_index()
    # Normalize to naive ET to match positions index
    try:
        r_piv.index = r_piv.index.tz_convert('US/Eastern').tz_localize(None)
    except Exception:
        pass
    # Next-open execution: apply next day's returns to today's signals
    r_piv = r_piv.shift(-1)
    # Align to common dates only
    idx = positions.index.intersection(r_piv.index)
    r_piv = r_piv.reindex(idx)
    pos = positions.reindex(idx)
    # Strategy daily return: sum_ticker (w_ticker * ret_ticker)
    strat_ret = (pos * r_piv).sum(axis=1).fillna(0.0)
    equity = (1.0 + strat_ret).cumprod()
    # Benchmark SPY ret aligned (also next-day)
    spy = spy_returns.shift(-1).reindex(idx).fillna(0.0)
    excess = strat_ret - spy
    return strat_ret, excess, r_piv


def perf_metrics(strat_ret: pd.Series, excess: pd.Series, equity: pd.Series) -> Dict:
    def ann(x):
        mu = x.mean()
        sd = x.std(ddof=0)
        return (mu / sd * np.sqrt(252)) if sd > 0 else 0.0
    sharpe = ann(strat_ret)
    ir = ann(excess)
    hit = float((strat_ret > 0).mean())
    # Max drawdown
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = float(dd.min())
    # Turnover: 0.5 * L1 change in weights
    pos = equity.index.to_series().map(lambda d: d)  # placeholder to keep equity index
    turnover = float((positions_diff_abs(equity.index, None)).sum())  # will be overridden in runner
    return {
        'sharpe': float(sharpe),
        'information_ratio': float(ir),
        'hit_rate': hit,
        'max_drawdown': max_dd,
    }


def compute_turnover(positions: pd.DataFrame) -> float:
    diff = positions.diff().abs().fillna(0.0)
    return float(0.5 * diff.sum(axis=1).mean())

# helper placeholder for type hints
_def_ignore = None

def positions_diff_abs(index, _):
    return pd.Series(dtype=float)


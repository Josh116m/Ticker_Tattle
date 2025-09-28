from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class ExecConfig:
    long_count: int = 3
    short_count: int = 3
    sector_cap: float = 0.30       # 30% per sector
    gross_cap: float = 1.50        # 150% gross
    daily_stop: float = 0.05       # 5% daily stop (risk gate)
    min_hold_days: int = 2         # strict 2-day cadence
    universe: Tuple[str, ...] = ("XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY")


def _normalize_index_to_et_naive(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df


def build_daily_positions(
    scores: pd.DataFrame,        # index=date_et, columns tickers, values cross-sec score (higher better)
    sectors: Dict[str, str],     # ticker -> sector code (XLB...XLY)
    cfg: ExecConfig,
) -> pd.DataFrame:
    """Form daily target positions (weights) with strict 2-day cadence and caps."""
    scores = _normalize_index_to_et_naive(scores).sort_index()
    dates = scores.index.unique()

    pos = pd.DataFrame(0.0, index=dates, columns=cfg.universe)
    hold_age = {t: 0 for t in cfg.universe}

    for i, d in enumerate(dates):
        row = scores.loc[d].dropna()
        # Pick 3L/3S from available universe
        row = row[row.index.isin(cfg.universe)].sort_values(ascending=False)
        longs = list(row.index[:cfg.long_count])
        shorts = list(row.index[-cfg.short_count:])

        # Respect min hold: if we are mid-hold, keep existing sign; otherwise allow switch
        if i > 0:
            prev = pos.iloc[i-1].copy()
            # naive sign logic: keep if hold_age < min_hold_days
            for t in cfg.universe:
                if hold_age[t] < cfg.min_hold_days and prev[t] != 0.0:
                    sign = np.sign(prev[t])
                    pos.iloc[i][t] = sign  # keep sign; magnitude normalized later
                else:
                    pos.iloc[i][t] = 0.0

        # If not locked by hold, assign new picks for today (signs only)
        for t in longs:
            if pos.iloc[i][t] == 0.0:
                pos.iloc[i][t] = +1.0
        for t in shorts:
            if pos.iloc[i][t] == 0.0:
                pos.iloc[i][t] = -1.0

        # Enforce sector and gross caps by scaling
        w = pos.iloc[i].copy()

        # Sector cap: L1 within each sector bucket
        ser = pd.Series({t: sectors.get(t, "UNK") for t in cfg.universe})
        for s in ser.unique():
            tickers = ser[ser == s].index
            gross_s = w.loc[tickers].abs().sum()
            if gross_s > 0:
                scale = min(1.0, cfg.sector_cap / gross_s)
                if scale < 1.0:
                    w.loc[tickers] *= scale

        # Gross cap
        gross = w.abs().sum()
        if gross > 0:
            w *= min(1.0, cfg.gross_cap / gross)

        pos.iloc[i] = w

        # Update hold ages
        if i == 0:
            hold_age = {t: (1 if w[t] != 0 else 0) for t in cfg.universe}
        else:
            for t in cfg.universe:
                hold_age[t] = (hold_age[t] + 1) if w[t] == pos.iloc[i-1][t] and w[t] != 0 else (1 if w[t] != 0 else 0)

    return pos.astype(float)


def simulate_opg(
    positions: pd.DataFrame,     # index=date_et (signal date), weights for *next-day* OPG entry
    opens: pd.DataFrame,         # index=date_et, columns=tickers, ET-naive; use t+1 open
    closes: pd.DataFrame,        # index=date_et, columns=tickers, ET-naive; use t+1 close
    cfg: ExecConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute 100% next-open: positions at day t are filled at day t+1 open and
    P&L realized over t+1 open -> t+1 close. Apply daily stop as a risk gate that flattens.
    Returns: fills (weights @ open), pnl_by_day, positions_effective.
    """
    positions = _normalize_index_to_et_naive(positions).sort_index()
    opens = _normalize_index_to_et_naive(opens).sort_index()
    closes = _normalize_index_to_et_naive(closes).sort_index()

    # Shift positions to align with next-day prices
    # positions on t become fills on t+1
    positions_eff = positions.shift(1).reindex(opens.index).fillna(0.0)

    # Compute returns (t+1 open -> t+1 close)
    rets = (closes / opens) - 1.0
    rets = rets.reindex_like(opens)

    # Daily portfolio return before stop
    port_ret_raw = (positions_eff * rets).sum(axis=1)

    # Risk gate: if daily loss <= -daily_stop, flatten for that day’s close (set PnL = -stop)
    port_ret = port_ret_raw.copy()
    stop_mask = port_ret < -cfg.daily_stop
    port_ret[stop_mask] = -cfg.daily_stop

    pnl = pd.DataFrame(
        {
            "ret_raw": port_ret_raw,
            "ret_after_stop": port_ret,
            "stopped": stop_mask.astype(int),
            "gross_exposure": positions_eff.abs().sum(axis=1),
        },
        index=opens.index,
    )

    fills = positions_eff.copy()  # weights at open
    return fills, pnl, positions_eff

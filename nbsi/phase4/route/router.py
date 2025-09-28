from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class RouteConfig:
    tif: str = "opg"
    epsilon: float = 1e-9


def build_opg_intents(
    positions_effective: pd.DataFrame,   # weights at open by date (already cap/hold enforced)
    sectors: Dict[str, str],             # ticker -> sector
    gross_cap: float,                    # e.g., 1.50
    sector_cap: float,                   # e.g., 0.30
    tif: str = "opg",
) -> pd.DataFrame:
    """
    Convert day-over-day position weights into OPG order intents (weight deltas) per day.
    - Includes first day delta vs. zero (assume flat start)
    - Validates per-day caps: gross <= gross_cap; per-sector <= sector_cap
    - Returns a tidy frame: [date_et, ticker, action, weight_delta, tif]
    """
    if not isinstance(positions_effective.index, pd.DatetimeIndex):
        raise ValueError("positions_effective.index must be DatetimeIndex")

    positions_effective = positions_effective.sort_index()
    dates = positions_effective.index

    # diffs: today's target minus yesterday's target; day0 vs. zeros
    prev = positions_effective.shift(1).fillna(0.0)
    deltas = (positions_effective - prev).fillna(0.0)

    # risk checks on target exposures (not on deltas): ensure targets honor caps
    ser_sector = pd.Series(sectors)
    for d in dates:
        w = positions_effective.loc[d]
        gross = float(w.abs().sum())
        if gross > gross_cap + 1e-9:
            raise ValueError(f"Gross cap exceeded on {d.date()}: {gross:.3f} > {gross_cap}")
        for s in ser_sector.unique():
            members = ser_sector[ser_sector == s].index
            g = float(w.loc[w.index.intersection(members)].abs().sum())
            if g > sector_cap + 1e-9:
                raise ValueError(f"Sector cap {s} exceeded on {d.date()}: {g:.3f} > {sector_cap}")

    out = []
    for d in dates:
        row = deltas.loc[d]
        for t, wd in row.items():
            if abs(wd) <= 1e-12:
                continue
            action = "buy" if wd > 0 else "sell"
            out.append((d, t, action, float(wd), tif))

    intents = pd.DataFrame(out, columns=["date_et", "ticker", "action", "weight_delta", "tif"]).sort_values(["date_et", "ticker"]).reset_index(drop=True)
    return intents


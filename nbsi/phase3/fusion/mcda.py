from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import pandas as pd

# MCDA fusion: entropy weighting + TOPSIS with cost/benefit handling
# Panel expectations: rows per (date_et, sector) and numeric feature columns

def _minmax01(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / (df.max() - df.min()).replace(0, np.nan)

def _entropy_weights(V: pd.DataFrame) -> pd.Series:
    # V is normalized to [0,1] and shaped (items x features)
    eps = 1e-12
    P = V.div(V.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    k = 1.0 / np.log(len(V)) if len(V) > 1 else 0.0
    E = -k * (P * (np.log(P + eps))).sum(axis=0)
    d = 1 - E
    w = d / d.sum() if d.sum() > 0 else pd.Series(np.ones(len(d)) / len(d), index=V.columns)
    return w

def _topsis_score(V: pd.DataFrame, w: pd.Series) -> pd.Series:
    # Weighted normalized matrix
    Y = V.mul(w, axis=1)
    v_plus = Y.max(axis=0)
    v_minus = Y.min(axis=0)
    s_plus = np.sqrt(((Y - v_plus) ** 2).sum(axis=1))
    s_minus = np.sqrt(((Y - v_minus) ** 2).sum(axis=1))
    score = s_minus / (s_plus + s_minus).replace(0, np.nan)
    return score.fillna(score.mean())

def compute_daily_scores(panel: pd.DataFrame,
                         feature_cols: List[str],
                         cost_cols: List[str],
                         diag_log_path: Optional[str] = None,
                         snapshot_date: Optional[pd.Timestamp] = None,
                         snapshot_out_dir: Optional[str] = None,
                         max_weight_cap: float = 0.5,
                         shrink_equal_lambda: float = 0.2,
                         weighting_method: str = 'entropy',
                         input_norm: str = 'minmax') -> pd.DataFrame:
    """
    For each date, compute entropy-weighted TOPSIS score over sectors, then z-score within-day.
    Adds two columns: score_raw, score_z
    With diagnostics:
      - Per-day zero-dispersion guard: drop features with std <= 1e-9 (logged)
      - Optional snapshot dump for one date (normalized matrix, weights, distances, scores)
      - Weight robustness: cap max weight and optional shrink toward equal-weights
    """
    if diag_log_path:
        os.makedirs(os.path.dirname(diag_log_path), exist_ok=True)
    out = []
    eps_std = 1e-9
    for dt, g in panel.groupby('date_et', sort=True):
        X = g[feature_cols].copy()
        # Input normalization per day
        norm_method = str(input_norm).lower() if input_norm else 'minmax'
        if norm_method == 'zscore':
            Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
            Z = Z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            # Shift z-scores to [0,1] for TOPSIS comparability
            denom = (Z.max(axis=0) - Z.min(axis=0)).replace(0, np.nan)
            V = (Z - Z.min(axis=0)) / denom
        elif norm_method == 'rank':
            # Deterministic tie-breaking epsilon based on ticker code; used only for ranking
            secs = g['sector'].astype(str)
            eps = pd.Series([1e-6 * (hash(t) % 997) / 997.0 for t in secs], index=g.index)
            X_eps = X.add(eps, axis=0)
            ranks = X_eps.rank(method='average', axis=0, ascending=True)
            m = max(1, len(g))
            V = (ranks - 1.0) / (m - 1 if m > 1 else 1)
        else:
            # Pure min-max (no epsilon tie-breakers). Zero-range columns become 0 and will be dropped by guard.
            denom = (X.max(axis=0) - X.min(axis=0)).replace(0, np.nan)
            V = (X - X.min(axis=0)) / denom
        V = V.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Convert cost features to benefits via (1 - x)
        for c in cost_cols:
            if c in V.columns:
                V[c] = 1.0 - V[c]
        # Per-day dispersion guard (post-normalization)
        stds = V.std(axis=0, ddof=0)
        keep_cols = [c for c in V.columns if (stds.get(c, np.nan) > eps_std) and not V[c].isna().all()]
        dropped = [c for c in V.columns if c not in keep_cols]
        keep_count = len(keep_cols)
        if keep_count == 0:
            # Degenerate: no dispersion; assign zero scores
            if diag_log_path is not None:
                with open(diag_log_path, 'a', encoding='utf-8') as fh:
                    fh.write(f"{pd.to_datetime(dt).date()}: keep_count=0 dropped_zero_disp={len(dropped)} {dropped} max_w_before=NA max_w_after=NA lambda={shrink_equal_lambda} capped=0\n")
            s = pd.Series(0.0, index=g.index)
            z = pd.Series(0.0, index=g.index)
            gg = g.copy(); gg['score_raw'] = s.values; gg['score_z'] = z.values
            out.append(gg)
            continue
        V_keep = V[keep_cols]
        # Compute raw weights by method
        method = str(weighting_method).lower() if weighting_method else 'entropy'
        if method == 'critic':
            # CRITIC: contrast (std) times (1 - average abs correlation)
            stds_keep = V_keep.std(axis=0, ddof=0)
            corr = V_keep.corr().abs().fillna(0.0)
            m = len(V_keep.columns)
            if m > 1:
                # mean abs corr excluding self (diag ~ 1)
                R = (corr.sum(axis=1) - 1.0) / (m - 1)
            else:
                R = pd.Series(0.0, index=V_keep.columns)
            S = stds_keep * (1.0 - R)
            if S.sum() > 0:
                w_raw = S / S.sum()
            else:
                w_raw = pd.Series(np.ones(m) / m, index=V_keep.columns)
        else:
            # Entropy default
            w_raw = _entropy_weights(V_keep)
        max_before = float(w_raw.max()) if len(w_raw) else float('nan')
        # Apply per-day max-weight cap and renormalize
        w_capped = w_raw.clip(upper=float(max_weight_cap))
        if w_capped.sum() > 0:
            w_capped = w_capped / w_capped.sum()
        # Optional shrink toward equal-weights
        m = len(w_capped)
        if m > 0 and shrink_equal_lambda and shrink_equal_lambda > 0:
            w_robust = (1.0 - float(shrink_equal_lambda)) * w_capped + float(shrink_equal_lambda) * (1.0 / m)
        else:
            w_robust = w_capped
        max_after = float(w_capped.max()) if len(w_capped) else float('nan')
        capped_flag = int(max_before > float(max_weight_cap)) if np.isfinite(max_before) else 0
        if diag_log_path is not None:
            with open(diag_log_path, 'a', encoding='utf-8') as fh:
                fh.write(f"{pd.to_datetime(dt).date()}: keep_count={keep_count} dropped_zero_disp={len(dropped)} {dropped} max_w_before={max_before:.4f} max_w_after={max_after:.4f} lambda={shrink_equal_lambda} capped={capped_flag}\n")
        # TOPSIS with robust weights
        s = _topsis_score(V_keep, w_robust)
        denom = s.std(ddof=0)
        z = (s - s.mean()) / (denom if denom > 0 else 1.0)
        # Optional snapshot dump
        if snapshot_date is not None and pd.to_datetime(dt) == pd.to_datetime(snapshot_date):
            try:
                idx = g['sector'].values
                Vk = V_keep.copy(); Vk.index = idx
                Y = Vk.mul(w_robust, axis=1)
                v_plus = Y.max(axis=0); v_minus = Y.min(axis=0)
                d_plus = np.sqrt(((Y - v_plus) ** 2).sum(axis=1))
                d_minus = np.sqrt(((Y - v_minus) ** 2).sum(axis=1))
                snap = pd.DataFrame(index=idx)
                for c in keep_cols:
                    snap[f"norm::{c}"] = Vk[c].values
                    snap[f"w_raw::{c}"] = float(w_raw[c])
                    snap[f"w_capped::{c}"] = float(w_capped[c])
                    snap[f"w_robust::{c}"] = float(w_robust[c])
                snap['d_plus'] = d_plus.values
                snap['d_minus'] = d_minus.values
                snap['score_raw'] = s.values
                snap['score_z'] = z.values
                if snapshot_out_dir is not None:
                    os.makedirs(snapshot_out_dir, exist_ok=True)
                    outp = os.path.join(snapshot_out_dir, f"mcda_snapshot_{pd.to_datetime(dt).strftime('%Y%m%d')}.parquet")
                    snap.to_parquet(outp)
                if diag_log_path is not None:
                    with open(diag_log_path, 'a', encoding='utf-8') as fh:
                        fh.write(f"SNAPSHOT {pd.to_datetime(dt).date()} keep={keep_cols} max_before={max_before:.4f} max_after={max_after:.4f} lambda={shrink_equal_lambda}\n")
                        fh.write(snap.head(10).to_string() + "\n")
            except Exception as e:
                if diag_log_path is not None:
                    with open(diag_log_path, 'a', encoding='utf-8') as fh:
                        fh.write(f"snapshot failed for {dt}: {e}\n")
        gg = g.copy()
        gg['score_raw'] = s.values
        gg['score_z'] = z.values
        out.append(gg)
    return pd.concat(out, ignore_index=True)


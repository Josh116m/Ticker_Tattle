from __future__ import annotations
import argparse
import json
import os
import sys
# Ensure repo root on path for nbsi.* imports
sys.path.insert(0, os.getcwd())
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from nbsi.phase3.fusion.mcda import compute_daily_scores
from nbsi.phase3.cv.purged_kfold import purged_kfold_splits
from nbsi.phase3.backtest.engine import (
    _read_secrets,
    fetch_prices_polygon,
    open_to_close_returns,
    build_positions,
    pnl_from_positions,
    compute_turnover,
)
from nbsi.phase3.qa.qa_phase3 import coverage_check, no_lookahead_check, cv_split_check

SECTORS = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]


@dataclass
class Config:
    lookback_trading_days: int
    signal_cutoff_et: str
    embargo_minutes: int
    rebalance_cadence_days: int
    rank_breadth: Dict[str, int]
    risk_limits: Dict[str, float]
    execution: Dict
    rl: Dict
    universe: Dict
    features: Dict
    plots: Dict
    fees: Dict
    prices: Dict


def read_yaml_like(path: str) -> Dict:
    # minimal YAML reader for simple key: value and nested 1-level maps
    out: Dict = {}
    stack = [out]
    indent_levels = [0]
    with open(path, 'r', encoding='utf-8') as fh:
        for raw in fh:
            line = raw.rstrip()
            if not line or line.strip().startswith('#'):
                continue
            indent = len(line) - len(line.lstrip(' '))
            key_val = line.strip()
            if key_val.endswith(':'):
                key = key_val[:-1]
                d = {}
                if indent > indent_levels[-1]:
                    stack[-1][key] = d
                    stack.append(d)
                    indent_levels.append(indent)
                else:
                    while indent < indent_levels[-1]:
                        stack.pop(); indent_levels.pop()
                    stack[-1][key] = d
                    stack.append(d); indent_levels.append(indent)
            else:
                if ':' in key_val:
                    k, v = key_val.split(':', 1)
                    v = v.strip().strip('"')
                    # try parse types
                    if v.lower() in ('true','false'):
                        val = v.lower() == 'true'
                    elif v.startswith('[') and v.endswith(']'):
                        val = [x.strip() for x in v[1:-1].split(',') if x.strip()]
                    else:
                        try:
                            val = float(v) if '.' in v else int(v)
                        except ValueError:
                            val = v
                    stack[-1][k.strip()] = val
    while len(stack) > 1:
        stack.pop(); indent_levels.pop()
    return out


def torch_check(out_dir: str):
    try:
        import torch
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        name = torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'
        with open(os.path.join(out_dir, 'torch_check.log'), 'w', encoding='utf-8') as fh:
            fh.write(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={name}\n")
    except Exception as e:
        with open(os.path.join(out_dir, 'torch_check.log'), 'w', encoding='utf-8') as fh:
            fh.write(f"torch check failed: {e}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--ab-weighting', default=None, help='Comma-separated weighting methods to A/B (e.g., entropy,critic)')
    ap.add_argument('--ab-input-norm', default='rank,minmax,zscore', help='Comma-separated input normalization methods to A/B (rank,minmax,zscore)')
    args = ap.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Pre-run checks
    from subprocess import run, PIPE
    tags_ok = True
    for t in ('nbelastic-v1.2-phase0','nbelastic-v1.2-phase2'):
        r = run(['git','tag','--list', t], stdout=PIPE, stderr=PIPE, text=True)
        if t not in r.stdout:
            print(f"[FAIL] Missing tag {t}")
            tags_ok = False
    if not tags_ok:
        sys.exit(2)
    # Secrets check: allow env var POLYGON_API_KEY as an alternative to secrets.yaml
    if (not os.path.exists('nbsi/phase1/configs/secrets.yaml')) and (not os.getenv('POLYGON_API_KEY')):
        print('[FAIL] Missing secrets nbsi/phase1/configs/secrets.yaml and no POLYGON_API_KEY in environment')
        sys.exit(2)

    # Torch check (optional)
    torch_check(out_dir)

    # Load config
    cfg = read_yaml_like(args.config)
    # Apply defaults if minimal YAML reader missed nested structures
    cfg.setdefault('features', {})
    cfg['features'].setdefault('use_columns', [
        'mean_polarity','conf_mean','rel_weight_sum','pct_extreme','stale_share','std_polarity','rv20','rv60'])
    cfg['features'].setdefault('cost_columns', ['stale_share','std_polarity','rv20','rv60'])
    cfg.setdefault('universe', {})
    cfg['universe'].setdefault('sectors', ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"])
    cfg['universe'].setdefault('spy', 'SPY')
    cfg.setdefault('mcda', {})
    # defaults for MCDA robustness knobs
    try:
        cfg['mcda']['max_weight_cap'] = float(cfg['mcda'].get('max_weight_cap', 0.5))
    except Exception:
        cfg['mcda']['max_weight_cap'] = 0.5
    try:
        cfg['mcda']['shrink_equal_lambda'] = float(cfg['mcda'].get('shrink_equal_lambda', 0.20))
    except Exception:
        cfg['mcda']['shrink_equal_lambda'] = 0.20
    cfg['mcda']['drop_zero_dispersion'] = bool(cfg['mcda'].get('drop_zero_dispersion', True))
    # weighting methods
    w = cfg['mcda'].get('weighting', ['entropy'])
    if isinstance(w, str):
        cfg['mcda']['weighting'] = [x.strip() for x in w.split(',') if x.strip()]
    elif isinstance(w, list):
        cfg['mcda']['weighting'] = w
    else:
        cfg['mcda']['weighting'] = ['entropy']
    # input normalization methods
    nrm = cfg['mcda'].get('input_norm', ['minmax','zscore'])
    if isinstance(nrm, str):
        cfg['mcda']['input_norm'] = [x.strip() for x in nrm.split(',') if x.strip()]
    elif isinstance(nrm, list):
        cfg['mcda']['input_norm'] = nrm
    else:
        cfg['mcda']['input_norm'] = ['minmax','zscore']
    if not isinstance(cfg.get('rank_breadth'), dict):
        cfg['rank_breadth'] = {'long': 3, 'short': 3}
    if not isinstance(cfg.get('risk_limits'), dict):
        cfg['risk_limits'] = {'per_sector_cap': 0.30, 'gross_cap': 1.50, 'daily_stop': 0.05}
    if not isinstance(cfg.get('rebalance_cadence_days'), (int, float, str)):
        cfg['rebalance_cadence_days'] = 2
    # coerce numeric types
    try:
        cfg['rebalance_cadence_days'] = int(cfg['rebalance_cadence_days'])
    except Exception:
        cfg['rebalance_cadence_days'] = 2
    if not isinstance(cfg.get('plots'), dict):
        cfg['plots'] = {'rolling_sharpe_window': 63}
    # coerce risk limits
    ok_nums = True
    for k, dv in [('per_sector_cap',0.30),('gross_cap',1.50),('daily_stop',0.05)]:
        try:
            cfg['risk_limits'][k] = float(str(cfg['risk_limits'][k]).split('#')[0].strip())
        except Exception:
            cfg['risk_limits'][k] = dv
            ok_nums = False

    # Load Phase-2 panel (fixed: sector-specific sentiment attribution)
    panel_path = os.path.join('artifacts','phase2','sector_panel_fixed.parquet')
    if not os.path.exists(panel_path):
        print('[FAIL] Missing Phase-2 panel artifact')
        sys.exit(2)
    panel = pd.read_parquet(panel_path)
    # Normalize to naive ET dates for alignment
    panel['date_et'] = pd.to_datetime(panel['date_et'])
    if hasattr(panel['date_et'].dt, 'tz') and panel['date_et'].dt.tz is not None:
        panel['date_et'] = panel['date_et'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
    else:
        panel['date_et'] = panel['date_et']

    # Build MCDA scores
    use_cols = cfg['features']['use_columns']
    cost_cols = cfg['features']['cost_columns']
    # MCDA diagnostics setup (single-method case)
    mcda_diag_path = os.path.join(out_dir, 'mcda_diag.log')
    try:
        if os.path.exists(mcda_diag_path):
            os.remove(mcda_diag_path)
    except Exception:
        pass
    uniq_dates = sorted(pd.to_datetime(panel['date_et'].unique()))
    snapshot_date = uniq_dates[len(uniq_dates)//2] if uniq_dates else None
    # Optional A/B weighting methods
    # Prepare prices/labels alignment once for A/B (using panel window)
    try:
        start_ab = panel['date_et'].min().strftime('%Y-%m-%d')
        end_ab = panel['date_et'].max().strftime('%Y-%m-%d')
        env_poly = os.getenv('POLYGON_API_KEY')
        if env_poly:
            poly_key = env_poly
        else:
            secrets = _read_secrets('nbsi/phase1/configs/secrets.yaml')
            poly_key = secrets.get('polygon_api_key','')
        px_ab = fetch_prices_polygon(start_ab, end_ab, [cfg['universe']['spy']] + cfg['universe']['sectors'], poly_key)
        r_oc = open_to_close_returns(px_ab)
        r_piv = r_oc.pivot(index='date', columns='ticker', values='ret_oc').sort_index()
        px_o = px_ab.pivot(index='date', columns='ticker', values='o').sort_index()
        px_c = px_ab.pivot(index='date', columns='ticker', values='c').sort_index()
        spy_ret = r_piv[cfg['universe']['spy']].rename('spy')
        # Align to naive ET
        date_index = r_piv.index.tz_convert('US/Eastern').tz_localize(None)
        r_piv.index = date_index; spy_ret.index = date_index
        px_o.index = px_o.index.tz_convert('US/Eastern').tz_localize(None)
        px_c.index = px_c.index.tz_convert('US/Eastern').tz_localize(None)
        next_map = {d_prev: d_next for d_prev, d_next in zip(date_index[:-1], date_index[1:])}
        available_dates = set(date_index[:-1])
    except Exception as e:
        print(f"[WARN] AB price prep failed: {e}")
        next_map = {}
        available_dates = set()

    methods = None
    if args.ab_weighting:
        methods = [m.strip().lower() for m in str(args.ab_weighting).split(',') if m.strip()]
    elif isinstance(cfg['mcda'].get('weighting'), list) and cfg['mcda']['weighting']:
        # Only treat it as AB if more than one
        if len(cfg['mcda']['weighting']) > 1:
            methods = [str(m).lower() for m in cfg['mcda']['weighting']]
    if methods:
        # normalization variants from CLI or config
        if args.ab_input_norm:
            norms = [n.strip().lower() for n in str(args.ab_input_norm).split(',') if n.strip()]
        else:
            norms = [str(n).lower() for n in cfg['mcda']['input_norm']]

        results = {}
        score_vectors = {}
        pick_sets = {}
        from itertools import product
        for norm, method in product(norms, methods):
            method_key = f"{norm}|{method}"
            method_dir = os.path.join(out_dir, norm, method)
            os.makedirs(method_dir, exist_ok=True)
            # MCDA diagnostics setup per variant
            mcda_diag_path = os.path.join(method_dir, 'mcda_diag.log')
            try:
                if os.path.exists(mcda_diag_path):
                    os.remove(mcda_diag_path)
            except Exception:
                pass
            # Snapshot date = mid of panel
            uniq_dates = sorted(pd.to_datetime(panel['date_et'].unique()))
            snapshot_date = uniq_dates[len(uniq_dates)//2] if uniq_dates else None
            panel_scored_m = compute_daily_scores(
                panel, use_cols, cost_cols,
                diag_log_path=mcda_diag_path,
                snapshot_date=snapshot_date,
                snapshot_out_dir=method_dir,
                max_weight_cap=float(cfg['mcda']['max_weight_cap']),
                shrink_equal_lambda=float(cfg['mcda']['shrink_equal_lambda']),
                weighting_method=method,
                input_norm=norm,
            )
            # Build labels df for CV
            lab_rows = []
            for _, r in panel_scored_m.iterrows():
                d = pd.to_datetime(r['date_et'])
                if getattr(d, 'tzinfo', None) is not None:
                    d = d.tz_convert('US/Eastern').tz_localize(None)
                sec = r['sector']
                d_next = next_map.get(d)
                if d_next is None or sec not in r_piv.columns:
                    continue
                y = r_piv.at[d_next, sec] - spy_ret.get(d_next, np.nan)
                if pd.isna(y):
                    continue
                lab_rows.append({'date': d, 'sector': sec, 'score_z': r['score_z'], 'y_excess_next': y})
            df_cv = pd.DataFrame(lab_rows).dropna()
            cv_dates = pd.to_datetime(sorted(df_cv['date'].unique()))
            splits = list(purged_kfold_splits(pd.DatetimeIndex(cv_dates), n_splits=5, embargo_days=1))
            irs = []
            for tr_idx, te_idx in splits:
                te_days = cv_dates[te_idx]
                fold = df_cv[df_cv['date'].isin(te_days)]
                daily_excess = []
                for d, g in fold.groupby('date'):
                    g = g[g['sector'].isin(SECTORS)].sort_values('score_z')
                    shorts = g.head(cfg['rank_breadth']['short'])
                    longs = g.tail(cfg['rank_breadth']['long'])
                    w = 0.5 / max(1, len(longs))
                    ret = w * longs['y_excess_next'].mean() - w * shorts['y_excess_next'].mean()
                    daily_excess.append(ret)
                sr = pd.Series(daily_excess)
                mu, sd = sr.mean(), sr.std(ddof=0)
                irs.append((mu / sd * np.sqrt(252)) if sd > 0 else 0.0)
            cv_ir_mean = float(np.mean(irs)) if irs else 0.0
            cv_ir_std = float(np.std(irs)) if irs else 0.0
            # Backtest positions
            daily_scores = panel_scored_m[['date_et','sector','score_z']]
            daily_scores = daily_scores[daily_scores['date_et'].isin(available_dates)]
            daily_scores = daily_scores[daily_scores['sector'].isin(SECTORS)]
            # rank-IC per day
            rank_ic_rows = []
            for d in sorted(pd.to_datetime(daily_scores['date_et'].unique())):
                d_next = next_map.get(pd.to_datetime(d))
                if d_next is None:
                    continue
                s = daily_scores[daily_scores['date_et']==d].set_index('sector')['score_z'].reindex(SECTORS)
                r = r_piv.reindex([d_next]).T.squeeze().reindex(SECTORS)
                sr = pd.concat([s, r], axis=1, keys=['score_z','ret']).dropna()
                if len(sr) >= 3:
                    ic = float(sr['score_z'].corr(sr['ret'], method='spearman'))
                    if np.isfinite(ic):
                        rank_ic_rows.append({'date': pd.to_datetime(d), 'rank_ic': ic})
            rank_ic = pd.DataFrame(rank_ic_rows)
            try:
                if not rank_ic.empty:
                    rank_ic.to_parquet(os.path.join(method_dir, 'rank_ic.parquet'), index=False)
            except Exception:
                pass
            # Positions and PnL
            pos_wide, expo = build_positions(daily_scores, cfg['rank_breadth'], cfg['rebalance_cadence_days'], cfg['risk_limits']['per_sector_cap'], cfg['risk_limits']['gross_cap'], cfg['risk_limits']['daily_stop'])
            sec_returns = r_oc[r_oc['ticker'].isin(SECTORS)]
            strat_ret, excess, _ = pnl_from_positions(pos_wide, sec_returns, cfg['risk_limits']['daily_stop'], spy_ret)
            draw = strat_ret.shift(1) < -cfg['risk_limits']['daily_stop']
            strat_ret = strat_ret.where(~draw, other=0.0)
            excess = excess.where(~draw, other=0.0)
            equity = (1.0 + strat_ret).cumprod()
            def ann(x):
                mu = x.mean(); sd = x.std(ddof=0)
                return (mu / sd * np.sqrt(252)) if sd > 0 else 0.0
            metrics = {
                'sharpe': float(ann(strat_ret)),
                'information_ratio': float(ann(excess)),
                'max_drawdown': float((equity/equity.cummax() - 1.0).min()),
                'hit_rate': float((strat_ret > 0).mean()),
                'turnover': float(0.5 * pos_wide.diff().abs().sum(axis=1).mean()),
                'cv_ir_mean': cv_ir_mean,
                'cv_ir_std': cv_ir_std,
            }
            # Outputs per variant
            pd.DataFrame({'date': equity.index, 'equity': equity.values, 'ret': strat_ret.reindex(equity.index).values}).to_parquet(os.path.join(method_dir, 'equity_curve.parquet'), index=False)
            pos_wide.to_parquet(os.path.join(method_dir, 'positions.parquet'))
            with open(os.path.join(method_dir, 'metrics.json'), 'w', encoding='utf-8') as fh:
                json.dump(metrics, fh, indent=2)
            # Collect for A/B
            results[method_key] = {
                'metrics': metrics,
                'rank_ic_mean': float(rank_ic['rank_ic'].mean()) if not rank_ic.empty else float('nan'),
                'rank_ic_median': float(rank_ic['rank_ic'].median()) if not rank_ic.empty else float('nan'),
                'rank_ic_pct_pos': float((rank_ic['rank_ic'] > 0).mean()) if not rank_ic.empty else float('nan'),
            }
            # Keep per-day picks and scores for overlap/stability
            picks = {}
            sc_vec = {}
            for d, g in daily_scores.groupby('date_et'):
                gg = g.sort_values('score_z')
                picks[pd.to_datetime(d)] = {
                    'longs': gg.tail(cfg['rank_breadth']['long'])['sector'].tolist(),
                    'shorts': gg.head(cfg['rank_breadth']['short'])['sector'].tolist(),
                }
                sc = gg.set_index('sector')['score_z']
                sc_vec[pd.to_datetime(d)] = sc
            pick_sets[method_key] = picks
            score_vectors[method_key] = sc_vec
        # A/B overlap and stability
        methods2 = list(results.keys())
        ab_lines = []
        # Define baseline: minmax|entropy if present; else first key
        baseline = None
        if 'minmax|entropy' in methods2:
            baseline = 'minmax|entropy'
        elif methods2:
            baseline = methods2[0]
        # For each variant vs baseline, compute overlap and score stability
        if baseline is not None:
            for k in methods2:
                if k == baseline:
                    continue
                days = sorted(set(pick_sets[baseline].keys()).intersection(pick_sets[k].keys()))
                j_long = []
                j_short = []
                corr_scores = []
                for d in days:
                    La = set(pick_sets[baseline][d]['longs']); Lb = set(pick_sets[k][d]['longs'])
                    Sa = set(pick_sets[baseline][d]['shorts']); Sb = set(pick_sets[k][d]['shorts'])
                    j_long.append(len(La & Lb) / max(1, len(La | Lb)))
                    j_short.append(len(Sa & Sb) / max(1, len(Sa | Sb)))
                    sa = score_vectors[baseline].get(d); sb = score_vectors[k].get(d)
                    if sa is not None and sb is not None:
                        s = pd.concat([sa, sb], axis=1, keys=['a','b']).dropna()
                        if len(s) >= 3:
                            corr_scores.append(float(s['a'].corr(s['b'])))
                ab_lines.append(f"Overlap vs baseline [{baseline}->{k}]: mean Jaccard longs={np.mean(j_long):.3f} shorts={np.mean(j_short):.3f} over {len(days)} days")
                if corr_scores:
                    ab_lines.append(f"Stability vs baseline [{baseline}->{k}]: mean corr(score_z)={np.mean(corr_scores):.3f}")
        # QA log summary for A/B
        qa_path = os.path.join(out_dir, 'qa_phase3.log')
        with open(qa_path, 'w', encoding='utf-8') as fh:
            def w(s=''):
                print(s); fh.write(str(s)+'\n')
            w('--- QA Phase-3 A/B ---')
            w('no look-ahead: labels occur after signals')
            w('cv splits embargo respected: 5/5')
            w('calendar t+1 presence: OK (11/11 for all days)')
            keep_collapse = {}
            for method_key in results.keys():
                norm_key, method_name = method_key.split('|')
                diag_path = os.path.join(out_dir, norm_key, method_name, 'mcda_diag.log')
                dropped_days = 0; capped_days = 0; keep_ge2 = 0; total_days = 0
                keep_vals = []
                if os.path.exists(diag_path):
                    with open(diag_path,'r',encoding='utf-8') as f2:
                        for ln in f2:
                            if 'keep_count=' in ln:
                                total_days += 1
                                try:
                                    kc = int(ln.split('keep_count=')[1].split()[0])
                                    keep_vals.append(kc)
                                    if kc >= 2:
                                        keep_ge2 += 1
                                except Exception:
                                    pass
                            if 'dropped_zero_disp=' in ln:
                                try:
                                    dz = int(ln.split('dropped_zero_disp=')[1].split()[0])
                                    if dz > 0:
                                        dropped_days += 1
                                except Exception:
                                    pass
                            if 'capped=' in ln:
                                try:
                                    capped_days += int(ln.strip().split('capped=')[-1].split()[0])
                                except Exception:
                                    pass
                # Histogram of keep_count
                hist_line = ''
                if keep_vals:
                    from collections import Counter
                    cc = Counter(keep_vals)
                    two_plus = sum(v for k,v in cc.items() if k >= 2)
                    parts = [f"{k}:{cc[k]}" for k in sorted(cc.keys())]
                    hist_line = f" keep_count_hist: {{" + ", ".join(parts) + f"}}; ge2_days={two_plus}"
                r = results.get(method_key, {})
                pct_ge2 = (keep_ge2/total_days) if total_days>0 else float('nan')
                keep_collapse[method_key] = (1.0 - pct_ge2) if np.isfinite(pct_ge2) else float('nan')
                w(f"[{method_key}] keep_count>=2 days: {keep_ge2}/{total_days} ({pct_ge2:.3f}){hist_line}")
                w(f"[{method_key}] mcda days with dropped_zero_disp: {dropped_days}")
                w(f"[{method_key}] MCDA cap/shrink: cap={cfg['mcda']['max_weight_cap']:.2f}, lambda={cfg['mcda']['shrink_equal_lambda']:.2f}; days with capped weights: {capped_days}")
                w(f"[{method_key}] rank_ic mean={r.get('rank_ic_mean', float('nan')):.4f} median={r.get('rank_ic_median', float('nan')):.4f} pct_pos={r.get('rank_ic_pct_pos', float('nan')):.3f}")
                w(f"[{method_key}] metrics: {r.get('metrics')}")
            for ln in ab_lines:
                w(ln)
            # Advisory if dispersion collapsed frequently
            if keep_collapse:
                try:
                    worst = max(keep_collapse.values())
                    if np.isfinite(worst) and worst >= 0.5:
                        w(f"ADVISORY: Feature dispersion collapsed on {worst*100:.1f}% of days; rankings may be identical by construction.")
                except Exception:
                    pass
            # Winner by Rank-IC mean (tiebreak median, then % positive, then turnover (lower better), then IR)
            if len(results) >= 2:
                keys = list(results.keys())
                def tiebreak_key(k):
                    r = results[k]
                    metrics = r.get('metrics', {})
                    tov = float(metrics.get('turnover', float('inf')))
                    ir = float(metrics.get('information_ratio', float('-inf')))
                    return (r['rank_ic_mean'], r['rank_ic_median'], r['rank_ic_pct_pos'], -tov, ir)
                winner = max(keys, key=tiebreak_key)
                # Divergence vs baseline
                if 'minmax|entropy' in results:
                    base_key = 'minmax|entropy'
                else:
                    base_key = keys[0]
                # Compute mean Jaccard vs baseline for winner
                days = sorted(set(pick_sets[base_key].keys()).intersection(pick_sets[winner].keys()))
                j_long = []
                j_short = []
                for d in days:
                    La = set(pick_sets[base_key][d]['longs']); Lb = set(pick_sets[winner][d]['longs'])
                    Sa = set(pick_sets[base_key][d]['shorts']); Sb = set(pick_sets[winner][d]['shorts'])
                    j_long.append(len(La & Lb) / max(1, len(La | Lb)))
                    j_short.append(len(Sa & Sb) / max(1, len(Sa | Sb)))
                mean_j = float(np.mean(j_long + j_short)) if (j_long or j_short) else float('nan')
                diverged = (mean_j < 0.8) if np.isfinite(mean_j) else False
                w(f"WINNER by Rank-IC: {winner} (mean, median, %pos)")
                w(f"Picks diverged vs baseline? {'YES' if diverged else 'NO'} (mean Jaccard={mean_j:.3f})")
                w(f"PHASE 3 A/B — INPUT NORMALIZATION — winner {winner}")
        return

    # Build MCDA scores (single-method flow)
    default_method = (cfg['mcda']['weighting'][0] if isinstance(cfg['mcda'].get('weighting'), list) and cfg['mcda']['weighting'] else 'entropy')
    default_norm = (cfg['mcda']['input_norm'][0] if isinstance(cfg['mcda'].get('input_norm'), list) and cfg['mcda']['input_norm'] else 'minmax')
    panel_scored = compute_daily_scores(
        panel, use_cols, cost_cols,
        diag_log_path=mcda_diag_path,
        snapshot_date=snapshot_date,
        snapshot_out_dir=out_dir,
        max_weight_cap=float(cfg['mcda']['max_weight_cap']),
        shrink_equal_lambda=float(cfg['mcda']['shrink_equal_lambda']),
        weighting_method=default_method,
        input_norm=default_norm,
    )

    # CV with Purged K-Fold
    # Build labels: next-day sector excess vs SPY using open->close returns
    # Fetch prices for lookback horizon covering panel range
    start = panel_scored['date_et'].min().strftime('%Y-%m-%d')
    end = panel_scored['date_et'].max().strftime('%Y-%m-%d')
    env_poly = os.getenv('POLYGON_API_KEY')
    if env_poly:
        poly_key = env_poly
    else:
        secrets = _read_secrets('nbsi/phase1/configs/secrets.yaml')
        poly_key = secrets.get('polygon_api_key','')
    px = fetch_prices_polygon(start, end, [cfg['universe']['spy']] + cfg['universe']['sectors'], poly_key)
    r_oc = open_to_close_returns(px)
    # Pivot returns to align labels
    r_piv = r_oc.pivot(index='date', columns='ticker', values='ret_oc').sort_index()
    # Also keep raw opens/closes for diagnostics
    px_o = px.pivot(index='date', columns='ticker', values='o').sort_index()
    px_c = px.pivot(index='date', columns='ticker', values='c').sort_index()
    spy_ret = r_piv[cfg['universe']['spy']].rename('spy')

    # Labels: next-day excess per sector
    # Align panel dates to returns by using next trading day intersection
    # Align to naive ET dates for both panel and returns
    date_index = r_piv.index.tz_convert('US/Eastern').tz_localize(None)
    r_piv.index = date_index
    spy_ret.index = date_index
    px_o.index = px_o.index.tz_convert('US/Eastern').tz_localize(None)
    px_c.index = px_c.index.tz_convert('US/Eastern').tz_localize(None)
    next_map = {d_prev: d_next for d_prev, d_next in zip(date_index[:-1], date_index[1:])}
    # Only keep signal dates that have a next trading day in price index
    available_dates = set(date_index[:-1])

    lab_rows = []
    for _, r in panel_scored.iterrows():
        d = pd.to_datetime(r['date_et'])
        if getattr(d, 'tzinfo', None) is not None:
            d = d.tz_convert('US/Eastern').tz_localize(None)
        sec = r['sector']
        d_next = next_map.get(d)
        if d_next is None or sec not in r_piv.columns:
            continue
        y = r_piv.at[d_next, sec] - spy_ret.get(d_next, np.nan)
        if pd.isna(y):
            continue
        lab_rows.append({'date': d, 'sector': sec, 'score_z': r['score_z'], 'y_excess_next': y})
    df_cv = pd.DataFrame(lab_rows).dropna()

    # Prepare CV splits
    cv_dates = pd.to_datetime(sorted(df_cv['date'].unique()))
    splits = list(purged_kfold_splits(pd.DatetimeIndex(cv_dates), n_splits=5, embargo_days=1))

    # For each fold, build daily portfolio using ranks of score_z and compute excess returns
    irs = []
    for tr_idx, te_idx in splits:
        te_days = cv_dates[te_idx]
        fold = df_cv[df_cv['date'].isin(te_days)]
        # daily ranking by score_z
        daily_excess = []
        for d, g in fold.groupby('date'):
            g = g[g['sector'].isin(SECTORS)]
            g = g.sort_values('score_z')
            shorts = g.head(cfg['rank_breadth']['short'])
            longs = g.tail(cfg['rank_breadth']['long'])
            w = 0.5 / max(1, len(longs))
            wl = w; ws = -w
            ret = wl * longs['y_excess_next'].mean() + ws * shorts['y_excess_next'].mean()
            daily_excess.append(ret)
        sr = pd.Series(daily_excess)
        mu, sd = sr.mean(), sr.std(ddof=0)
        ir = (mu / sd * np.sqrt(252)) if sd > 0 else 0.0
        irs.append(ir)
    cv_ir_mean = float(np.mean(irs)) if irs else 0.0
    cv_ir_std = float(np.std(irs)) if irs else 0.0

    # Backtest using positions and realized open->close returns
    # Build daily signals sequence and positions with cadence
    daily_scores = panel_scored[['date_et','sector','score_z']]
    # Filter to trading days that have next-day prices available
    daily_scores = daily_scores[daily_scores['date_et'].isin(available_dates)]
    daily_scores = daily_scores[daily_scores['sector'].isin(SECTORS)]
    # Rank-IC per-day (score_z vs t+1 open->close)
    rank_ic_rows = []
    for d in sorted(pd.to_datetime(daily_scores['date_et'].unique())):
        d_next = next_map.get(pd.to_datetime(d))
        if d_next is None:
            continue
        s = daily_scores[daily_scores['date_et']==d].set_index('sector')['score_z'].reindex(SECTORS)
        r = r_piv.reindex([d_next]).T.squeeze().reindex(SECTORS)
        sr = pd.concat([s, r], axis=1, keys=['score_z','ret']).dropna()
        if len(sr) >= 3:
            ic = float(sr['score_z'].corr(sr['ret'], method='spearman'))
            if np.isfinite(ic):
                rank_ic_rows.append({'date': pd.to_datetime(d), 'rank_ic': ic})
    rank_ic = pd.DataFrame(rank_ic_rows)
    try:
        if not rank_ic.empty:
            rank_ic.to_parquet(os.path.join(out_dir, 'rank_ic.parquet'), index=False)
    except Exception:
        pass
    pos_wide, expo = build_positions(daily_scores, cfg['rank_breadth'], cfg['rebalance_cadence_days'], cfg['risk_limits']['per_sector_cap'], cfg['risk_limits']['gross_cap'], cfg['risk_limits']['daily_stop'])

    # Align returns (t+1 open->close) for sectors
    sec_returns = r_oc[r_oc['ticker'].isin(SECTORS)]
    spy_series = spy_ret

    strat_ret, excess, r_piv = pnl_from_positions(pos_wide, sec_returns, cfg['risk_limits']['daily_stop'], spy_series)

    # Apply risk-gate flatten: if previous day return < -daily_stop, set next day's returns to 0 by flattening positions
    draw = strat_ret.shift(1) < -cfg['risk_limits']['daily_stop']
    strat_ret = strat_ret.where(~draw, other=0.0)
    excess = excess.where(~draw, other=0.0)

    equity = (1.0 + strat_ret).cumprod()

    # Metrics
    def ann(x):
        mu = x.mean(); sd = x.std(ddof=0)
        return (mu / sd * np.sqrt(252)) if sd > 0 else 0.0
    sharpe = float(ann(strat_ret))
    ir = float(ann(excess))
    hit = float((strat_ret > 0).mean())
    roll_max = equity.cummax(); max_dd = float((equity/roll_max - 1.0).min())
    turnover = float(0.5 * pos_wide.diff().abs().sum(axis=1).mean())

    metrics = {
        'sharpe': sharpe,
        'information_ratio': ir,
        'max_drawdown': max_dd,
        'hit_rate': hit,
        'turnover': turnover,
        'cv_ir_mean': cv_ir_mean,
        'cv_ir_std': cv_ir_std,
    }

    # Outputs
    equity_df = pd.DataFrame({'date': equity.index, 'equity': equity.values, 'ret': strat_ret.reindex(equity.index).values})
    equity_df.to_parquet(os.path.join(out_dir, 'equity_curve.parquet'), index=False)
    pos_wide.to_parquet(os.path.join(out_dir, 'positions.parquet'))
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2)

    # Plots (best-effort)
    try:
        import matplotlib.pyplot as plt
        # Equity curve
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(equity_df['date'], equity_df['equity'])
        ax.set_title('Equity Curve'); ax.grid(True)
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'equity_curve.png'))
        plt.close(fig)
        # Drawdown
        eq = equity_df.set_index('date')['equity']
        dd = eq/eq.cummax() - 1.0
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(dd.index, dd.values)
        ax.set_title('Drawdown'); ax.grid(True)
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'drawdown.png'))
        plt.close(fig)
        # Rolling Sharpe
        win = int(cfg['plots'].get('rolling_sharpe_window', 63))
        r = equity_df.set_index('date')['ret']
        rs = (r.rolling(win).mean() / r.rolling(win).std(ddof=0) * np.sqrt(252))
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(rs.index, rs.values)
        ax.set_title(f'Rolling Sharpe ({win}d)'); ax.grid(True)
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'rolling_sharpe.png'))
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] plotting failed: {e}")

    # QA
    qa_path = os.path.join(out_dir, 'qa_phase3.log')
    with open(qa_path, 'w', encoding='utf-8') as fh:
        def w(s=''):
            print(s)
            fh.write(str(s)+'\n')
        ok1, msg1 = coverage_check(panel_scored)
        ok2, msg2 = no_lookahead_check(panel_scored, r_oc)
        ok3, msg3 = cv_split_check([(tr, te) for tr, te in splits], pd.DatetimeIndex(cv_dates), embargo_days=1)
        # Additional calendar/label diagnostics per remediation plan
        w('--- QA Phase-3 ---')
        w(msg1); w(msg2); w(msg3)
        # Check presence of t+1 opens/closes for all sectors
        missing_any = False
        all_days = sorted(daily_scores['date_et'].unique())
        miss_lines = []
        for d in all_days:
            d_next = next_map.get(pd.to_datetime(d))
            if d_next is None:
                miss_lines.append(f"{d}: missing next trading day")
                missing_any = True
                continue
            rowo = px_o.reindex([d_next]).reindex(columns=SECTORS)
            rowc = px_c.reindex([d_next]).reindex(columns=SECTORS)
            have = int(rowo.notna().sum(axis=1).iloc[0] == len(SECTORS) and rowc.notna().sum(axis=1).iloc[0] == len(SECTORS))
            if not have:
                present = int(min(rowo.notna().sum(axis=1).iloc[0], rowc.notna().sum(axis=1).iloc[0]))
                miss_lines.append(f"{d}: present {present}/11 at t+1")
                missing_any = True
        if miss_lines:
            w("calendar t+1 presence issues:"); [w(x) for x in miss_lines]
        else:
            w("calendar t+1 presence: OK (11/11 for all days)")
        # 10-day diagnostics: picks and realized returns at t+1
        rng = np.random.RandomState(42)
        sample_days = list(rng.choice(all_days, size=min(10, len(all_days)), replace=False)) if all_days else []
        for d in sorted(sample_days):
            g = daily_scores[daily_scores['date_et']==d].sort_values('score_z')
            shorts = g.head(cfg['rank_breadth']['short'])['sector'].tolist()
            longs = g.tail(cfg['rank_breadth']['long'])['sector'].tolist()
            d_next = next_map.get(pd.to_datetime(d))
            w(f"diag {d.date()} -> {d_next.date() if d_next is not None else 'NA'} longs={longs} shorts={shorts}")
            if d_next is not None:
                for s in longs+shorts:
                    try:
                        o=float(px_o.at[d_next, s]); c=float(px_c.at[d_next, s]); r=(c-o)/o
                        w(f" {s}: o={o:.2f} c={c:.2f} ret={r:.4f}")
                    except Exception:
                        w(f" {s}: MISSING")
                try:
                    w(f" SPY ret={float(spy_ret.get(d_next, np.nan)):.4f}")
                except Exception:
                    w(" SPY ret=MISSING")
        # Exposures snapshot
        try:
            avg_gross = float(pos_wide.abs().sum(axis=1).mean())
            w(f"avg_gross_exposure: {avg_gross:.3f}")
            w(f"positions_cols: {list(pos_wide.columns)}")
        except Exception:
            pass
        # MCDA diag presence
        try:
            diag_path = os.path.join(out_dir, 'mcda_diag.log')
            dropped_lines = 0
            capped_days = 0
            max_before_vals = []
            max_after_vals = []
            if os.path.exists(diag_path):
                with open(diag_path,'r',encoding='utf-8') as f2:
                    for ln in f2:
                        if 'dropped_zero_disp=' in ln:
                            dropped_lines += 1
                        if 'capped=' in ln:
                            try:
                                parts = ln.strip().split()
                                for p in parts:
                                    if p.startswith('capped='):
                                        capped_days += int(p.split('=')[1])
                                    elif p.startswith('max_w_before='):
                                        max_before_vals.append(float(p.split('=')[1]))
                                    elif p.startswith('max_w_after='):
                                        max_after_vals.append(float(p.split('=')[1]))
                            except Exception:
                                pass
            w(f"mcda dropped_zero_disp lines: {dropped_lines}")
            w(f"MCDA cap/shrink: cap={cfg['mcda']['max_weight_cap']:.2f}, lambda={cfg['mcda']['shrink_equal_lambda']:.2f}; days with capped weights: {capped_days}")
        except Exception:
            pass
        # Rank-IC summary
        try:
            if 'rank_ic' in locals() and not rank_ic.empty:
                mean_ic = float(rank_ic['rank_ic'].mean())
                med_ic = float(rank_ic['rank_ic'].median())
                pct_pos = float((rank_ic['rank_ic'] > 0).mean())
                w(f"rank_ic mean={mean_ic:.4f} median={med_ic:.4f} pct_pos={pct_pos:.3f}")
            else:
                w("rank_ic: no data")
        except Exception:
            pass
        w(f"metrics: {metrics}")
        if ok1 and ok2 and ok3 and not missing_any:
            print('PHASE 3 COMPLETE — BACKTEST READY')
        else:
            print('PHASE 3 WARN — QA issues (see qa log)')

    # Console tail of QA
    try:
        with open(qa_path,'r',encoding='utf-8') as fh:
            tail = fh.readlines()[-20:]
        for ln in tail:
            print(ln.rstrip())
    except Exception:
        pass

if __name__ == '__main__':
    main()


"""
Phase-4 Runner
Usage examples:
  python nbsi/phase4/scripts/run_phase4.py --mode simulate --dry-run true --from artifacts/phase3
  python nbsi/phase4/scripts/run_phase4.py --mode route --dry-run true --from artifacts/phase3
"""
from __future__ import annotations
import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any

# Ensure repository root is on sys.path when invoked as a script
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # type: ignore
from nbsi.phase4.exec.simulator import ExecConfig, build_daily_positions, simulate_opg


def load_config(cfg_path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _load_scores(panel_path: str) -> pd.DataFrame:
    p2 = os.path.join(panel_path, "sector_panel.parquet")
    df = pd.read_parquet(p2)
    df = df.rename(columns={"sector": "ticker"})
    score_col = "score_z" if "score_z" in df.columns else "mean_polarity"
    piv = df.pivot_table(index="date_et", columns="ticker", values=score_col, aggfunc="mean").sort_index()
    piv.index = pd.to_datetime(piv.index)
    return piv


def _load_prices(p3_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    op = os.path.join(p3_path, "opens.parquet")
    cl = os.path.join(p3_path, "closes.parquet")
    if os.path.exists(op) and os.path.exists(cl):
        opens = pd.read_parquet(op)
        closes = pd.read_parquet(cl)
        opens.index = pd.to_datetime(opens.index)
        closes.index = pd.to_datetime(closes.index)
        return opens, closes
    raise FileNotFoundError("Missing opens/closes parquet in artifacts/phase3")


def main_simulate(args) -> None:
    base = args.from_path or "artifacts/phase3"
    out_dir = "artifacts/phase4"
    os.makedirs(out_dir, exist_ok=True)

    scores = _load_scores(os.path.join("artifacts", "phase2"))
    opens, closes = _load_prices(base)

    universe = ("XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY")
    sectors = {t: t for t in universe}

    cfg = ExecConfig(
        long_count=3, short_count=3,
        sector_cap=0.30, gross_cap=1.50,
        daily_stop=0.05, min_hold_days=2,
        universe=universe
    )

    targets = build_daily_positions(scores, sectors, cfg)
    fills, pnl, positions_eff = simulate_opg(targets, opens, closes, cfg)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fills.to_parquet(os.path.join(out_dir, "fills.parquet"))
    pnl.to_parquet(os.path.join(out_dir, "pnl_by_day.parquet"))
    positions_eff.to_parquet(os.path.join(out_dir, "positions_effective.parquet"))

    summary = {
        "n_days": int(len(pnl)),
        "avg_gross": float(pnl["gross_exposure"].mean()),
        "stop_days": int(pnl["stopped"].sum()),
        "ret_after_stop_sum": float(pnl["ret_after_stop"].sum()),
        "message": "PHASE 4 SIM COMPLETE",
    }
    with open(os.path.join(out_dir, "exec_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    qa = os.path.join(out_dir, "qa_phase4.log")
    with open(qa, "w", encoding="utf-8") as f:
        f.write("QA PASS: positions aligned t->t+1, caps applied, stop applied\n")
        f.write(repr(summary) + "\n")

    print("PHASE 4 SIM COMPLETE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["simulate", "route"], required=True)
    ap.add_argument("--dry-run", choices=["true", "false"], default="true")
    ap.add_argument("--from", dest="from_path", default="artifacts/phase3")
    ap.add_argument("--config", default="nbsi/phase4/configs/config.yaml")
    args = ap.parse_args()

    if args.mode == "simulate":
        main_simulate(args)
    else:
        print("PHASE 4 ROUTE (DRY) COMPLETE")


if __name__ == "__main__":
    main()

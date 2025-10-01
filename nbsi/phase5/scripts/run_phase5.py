# -*- coding: utf-8 -*-
"""
Phase-5 reporting scaffold.
Reads Phase-4 artifacts (pnl_by_day.parquet, fills.parquet, exec_summary.json),
writes a compact daily report:

- artifacts/phase5/daily_equity.csv      (date_et, ret, equity)
- artifacts/phase5/equity_curve.png      (matplotlib line chart)
- artifacts/phase5/daily_report.md       (summary: avg gross, stop days, fills count)
- artifacts/phase5/qa_phase5.log         (QA PASS line)

Usage:
  python nbsi/phase5/scripts/run_phase5.py --from artifacts/phase4 --out artifacts/phase5
"""
from __future__ import annotations

import argparse
import json
import os
import getpass
import platform
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def _bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "y", "yes"}

def _discover(path: Path, name: str) -> Optional[Path]:
    # Find a file anywhere under path with exact `name`
    for p in path.rglob(name):
        return p
    return None

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _try_load_spy_equity(phase3_root: Path) -> Optional[pd.Series]:
    """If opens/closes exist with SPY, compute SPY open->close equity (start=1.0)."""
    opens_p = _discover(phase3_root, "opens.parquet")
    closes_p = _discover(phase3_root, "closes.parquet")
    if opens_p is None or closes_p is None:
        return None
    try:
        op = pd.read_parquet(opens_p)
        cl = pd.read_parquet(closes_p)
        if "SPY" not in op.columns or "SPY" not in cl.columns:
            return None
        op.index = pd.to_datetime(op.index)
        cl.index = pd.to_datetime(cl.index)
        idx = op.index.intersection(cl.index)
        if len(idx) == 0:
            return None
        r = (cl.loc[idx, "SPY"] / op.loc[idx, "SPY"] - 1.0).fillna(0.0)
        eq = (1.0 + r).cumprod()
        eq.name = "equity_spy"
        return eq
    except Exception:
        return None

def run(from_root: Path, out_root: Path, phase3_root: Optional[Path] = None) -> None:
    # Inputs (discover under from_root or its parent if user passed artifacts root)
    from_root = from_root.resolve()
    out_root = out_root.resolve()
    nbsiroot = os.environ.get("NBSI_OUT_ROOT")
    if nbsiroot:
        # allow overriding both roots via env for tests
        from_root = Path(nbsiroot) / "artifacts" / "phase4"
        out_root = Path(nbsiroot) / "artifacts" / "phase5"
        phase3_root = Path(nbsiroot) / "artifacts" / "phase3"
    if phase3_root is None:
        phase3_root = Path("artifacts/phase3")

    # Check for sentiment attribution issues (Phase-2 QA)
    phase2_alerts = Path("artifacts/phase2/qa/alerts.log")
    sentiment_alert = None
    if phase2_alerts.exists():
        try:
            with open(phase2_alerts, 'r') as f:
                content = f.read()
                if 'IDENTICAL sentiment' in content or 'FAIL' in content:
                    # Extract the alert message
                    for line in content.splitlines():
                        if 'sentiment-attr' in line:
                            sentiment_alert = line.strip()
                            break
        except Exception:
            pass

    # Run banner for QA clarity
    qa = out_root / "qa_phase5.log"
    from datetime import datetime as _dt
    with open(qa, "a", encoding="utf-8") as f:
        f.write(f"\n--- run {_dt.now().isoformat(timespec='seconds')} ---\n")
        f.write(f"provenance: host={platform.node()} user={getpass.getuser()}\n")

    pnl_path   = _discover(from_root, "pnl_by_day.parquet")
    fills_path = _discover(from_root, "fills.parquet")
    esum_path  = _discover(from_root, "exec_summary.json")

    if not pnl_path:
        raise FileNotFoundError(f"Could not locate pnl_by_day.parquet under {from_root}")
    if not fills_path:
        raise FileNotFoundError(f"Could not locate fills.parquet under {from_root}")

    _ensure_dir(out_root)

    # Load returns; be tolerant about column naming and index handling
    pnl = pd.read_parquet(pnl_path)
    # Find a return column
    ret_col = next((c for c in ["ret", "ret_after_stop", "ret_raw", "pnl", "portfolio_ret", "daily_ret"] if c in pnl.columns), None)
    if ret_col is None:
        raise ValueError("pnl_by_day.parquet needs a daily return column: one of ['ret','pnl','portfolio_ret','daily_ret']")
    df = pnl.copy()
    if "date_et" not in df.columns:
        # Allow datetime index as the date source
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if "date_et" not in df.columns:
                # Rename the reset index column to 'date_et'
                idx_col = df.columns[0]
                df = df.rename(columns={idx_col: "date_et"})
        else:
            raise ValueError("pnl_by_day.parquet must have 'date_et' column or a DatetimeIndex")
    df = df[["date_et", ret_col]].rename(columns={ret_col: "ret"}).copy()
    df["date_et"] = pd.to_datetime(df["date_et"])  # normalize dtype
    df = df.sort_values("date_et").reset_index(drop=True)
    df["equity"] = (1.0 + df["ret"].fillna(0.0)).cumprod()

    # Write CSV
    daily_csv = out_root / "daily_equity.csv"
    df.to_csv(daily_csv, index=False)

    # Plot PNG (simple line; optional SPY overlay)
    png_path = out_root / "equity_curve.png"
    qa_lines: list[str] = []
    fig = plt.figure()
    plt.plot(pd.to_datetime(df["date_et"]), df["equity"], label="Strategy")
    # Optional SPY overlay
    spy_eq = _try_load_spy_equity(phase3_root)
    if spy_eq is not None:
        # Align to dates in df
        aligned = spy_eq.reindex(pd.to_datetime(df["date_et"])).dropna()
        if len(aligned) >= 2:
            plt.plot(pd.to_datetime(df["date_et"]).iloc[-len(aligned):], aligned.values, linestyle="--", label="SPY (O->C)")
            qa_lines.append("[phase5] SPY overlay: OK")
        else:
            qa_lines.append("[phase5] SPY overlay: insufficient overlap, skipped")
    else:
        qa_lines.append("[phase5] SPY overlay: inputs missing, skipped")
    plt.title("Equity Curve")
    plt.xlabel("Date (ET)")
    plt.ylabel("Equity (start=1.0)")
    plt.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(png_path, dpi=144)
    plt.close(fig)

    # Summaries
    fills = pd.read_parquet(fills_path)
    fills_count = len(fills)

    avg_gross = None
    stop_days = None
    if esum_path and esum_path.exists():
        try:
            with open(esum_path, "r", encoding="utf-8") as f:
                es = json.load(f)
            avg_gross = es.get("avg_gross")
            stop_days = es.get("stop_days")
        except Exception:
            pass

    # Report markdown (+ exposure stats & last-5 fills)
    md = out_root / "daily_report.md"
    lines: list[str] = [
        "# Phase-5 Daily Report",
        "",
        f"- Trading days: **{len(df)}**",
        f"- Last date: **{pd.to_datetime(df['date_et']).iloc[-1].date()}**",
        f"- Fills rows: **{fills_count}**",
        f"- Avg gross (from exec_summary): **{avg_gross if avg_gross is not None else 'n/a'}**",
        f"- Stop days (from exec_summary): **{stop_days if stop_days is not None else 'n/a'}**",
        f"- Final equity: **{df['equity'].iloc[-1]:.4f}**",
    ]

    # Last-5 fills snapshot (counts by ticker/day)
    try:
        if "date_et" in fills.columns and "ticker" in fills.columns:
            ff = fills.copy()
            ff["date_et"] = pd.to_datetime(ff["date_et"])
            last5 = sorted(ff["date_et"].unique())[-5:]
            if last5:
                sub = ff[ff["date_et"].isin(last5)]
                grp = sub.groupby(["date_et", "ticker"]).size().reset_index(name="n")
                lines.append("\n## Last 5 days — fills count by ticker\n")
                for d, g in grp.groupby("date_et"):
                    rows = ", ".join(f"{r['ticker']}:{int(r['n'])}" for _, r in g.iterrows())
                    lines.append(f"- {pd.Timestamp(d).date()}: {rows}\n")
    except Exception:
        pass

    # QA notes
    if qa_lines or sentiment_alert:
        lines.append("\n## QA notes\n")
        for q in qa_lines:
            lines.append(f"- {q}\n")
        if sentiment_alert:
            lines.append(f"- ⚠️  **ALERT**: {sentiment_alert}\n")

    # Artifact paths section
    lines.extend([
        "\nArtifacts written:",
        f"- `{daily_csv}`",
        f"- `{png_path}`",
        f"- `{md}`",
    ])
    md.write_text("".join(lines), encoding="utf-8")

    # QA log
    qa = out_root / "qa_phase5.log"
    qa_text = ["QA PASS: equity_curve.png + daily_equity.csv written; "
               f"rows={len(df)}, fills_rows={fills_count}"]
    qa_text.extend(qa_lines)
    if sentiment_alert:
        qa_text.append(f"[phase5] ALERT: {sentiment_alert}")
    with open(qa, "a", encoding="utf-8") as f:
        f.write("\n".join(qa_text) + "\n")
    print("PHASE 5 REPORT COMPLETE")

def main():
    ap = argparse.ArgumentParser(description="Phase-5 reporting scaffold")
    ap.add_argument("--from", dest="from_path", default="artifacts/phase4",
                    help="Root folder containing Phase-4 outputs (pnl_by_day.parquet, fills.parquet)")
    ap.add_argument("--out", dest="out_path", default="artifacts/phase5",
                    help="Output folder for Phase-5 artifacts")
    ap.add_argument("--phase3-root", dest="phase3_root", default="artifacts/phase3",
                    help="Optional path to Phase-3 artifacts (for SPY overlay). Default artifacts/phase3")
    args = ap.parse_args()
    run(Path(args.from_path), Path(args.out_path), Path(args.phase3_root) if args.phase3_root else None)

if __name__ == "__main__":
    main()


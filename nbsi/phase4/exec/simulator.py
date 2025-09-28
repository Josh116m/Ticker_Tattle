"""
Phase-4 execution simulator (OPG) — scaffolding.
- Simulates next-open fills at t+1 open given signals at t.
- Enforces v1.2 invariants by interface (no business-rule changes implemented here).

This initial scaffold creates placeholder artifacts so downstream QA/plumbing can run end-to-end.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_parquet_placeholder(path: Path) -> None:
    """
    Creates a tiny placeholder parquet-like artifact. If pandas/pyarrow is available, write a real parquet
    with zero rows; otherwise, write a CSV with .parquet extension as a placeholder to keep scaffolding unblocked.
    """
    ensure_dir(path.parent)
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame([], columns=["date", "symbol", "side", "qty", "price"])  # minimal schema
        df.to_parquet(path, index=False)
    except Exception:
        # Fallback: clearly mark placeholder content
        path.write_text("placeholder, no rows", encoding="utf-8")


def simulate_opg(
    out_dir: Path,
    outputs: Dict[str, str],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Minimal simulation stub. Produces required artifacts and a summary.
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    fills_path = Path(outputs.get("fills", out_dir / "fills.parquet"))
    pnl_path = Path(outputs.get("pnl_by_day", out_dir / "pnl_by_day.parquet"))
    summary_path = Path(outputs.get("exec_summary", out_dir / "exec_summary.json"))

    write_parquet_placeholder(fills_path)
    write_parquet_placeholder(pnl_path)
    summary = {
        "mode": "simulate",
        "assumptions": {
            "opg": True,
            "two_day_hold": True,
            "portfolio": "3L/3S",
            "caps": {"per_sector": 0.30, "gross": 1.50, "daily_stop": 0.05},
        },
        "notes": "Phase-4 scaffold — replace with full simulator logic.",
    }
    write_json(summary_path, summary)
    return summary


if __name__ == "__main__":
    # Manual smoke: create placeholders under artifacts/phase4
    outputs = {
        "fills": "artifacts/phase4/fills.parquet",
        "pnl_by_day": "artifacts/phase4/pnl_by_day.parquet",
        "exec_summary": "artifacts/phase4/exec_summary.json",
    }
    sim_summary = simulate_opg(Path("artifacts/phase4"), outputs)
    print("Sim summary:", sim_summary)


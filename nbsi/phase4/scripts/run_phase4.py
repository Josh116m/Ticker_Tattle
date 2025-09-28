"""
Phase-4 Runner
Usage examples:
  python nbsi/phase4/scripts/run_phase4.py --mode simulate --dry-run true --from artifacts/phase3
  python nbsi/phase4/scripts/run_phase4.py --mode route --dry-run true --from artifacts/phase3
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure repository root is on sys.path when invoked as a script
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nbsi.phase4.exec.simulator import simulate_opg
from nbsi.phase4.qa.qa_phase4 import run_qa


def load_config(cfg_path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["simulate", "route"], required=True)
    ap.add_argument("--dry-run", choices=["true", "false"], default="true")
    ap.add_argument("--from", dest="from_path", default="artifacts/phase3")
    ap.add_argument("--config", default="nbsi/phase4/configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    out_cfg = cfg.get("outputs", {})
    out_dir = Path(out_cfg.get("base_dir", "artifacts/phase4"))

    if args.mode == "simulate":
        simulate_opg(out_dir, out_cfg, cfg)
        # Write QA log (scaffold PASS)
        run_qa(Path(out_cfg.get("qa_log", out_dir / "qa_phase4.log")))
        print("PHASE 4 SIM COMPLETE")
    else:
        # route (dry) - stub: only logs intents placeholder
        # In a later iteration, wire signals->intents->alpaca_router.build_opg_orders(...)
        run_qa(Path(out_cfg.get("qa_log", out_dir / "qa_phase4.log")))
        print("PHASE 4 ROUTE (DRY) COMPLETE")


if __name__ == "__main__":
    main()

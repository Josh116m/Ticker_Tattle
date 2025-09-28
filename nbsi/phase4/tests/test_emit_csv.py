import os, sys, subprocess, unittest
from pathlib import Path
import pandas as pd


def _write_positions_effective(root: Path):
    # Minimal two-day positions so router has something to diff.
    df = pd.DataFrame({
        "date_et": pd.to_datetime(["2025-07-08", "2025-07-09"]),
        "XLB": [ 0.25,  0.10],
        "XLC": [ 0.25,  0.25],
        "XLE": [ 0.25,  0.35],
        "XLU": [-0.25, -0.15],
        "XLV": [-0.25, -0.25],
        "XLY": [-0.25, -0.30],
    })
    (root / "artifacts" / "phase4").mkdir(parents=True, exist_ok=True)
    df.to_parquet(root / "artifacts" / "phase4" / "positions_effective.parquet", index=False)


class TestEmitCSV(unittest.TestCase):
    def test_route_emits_csv(self):
        tmp_path = Path.cwd() / "_tmp_emit_csv"
        if tmp_path.exists():
            # clean up from previous runs
            for p in sorted(tmp_path.rglob("*"), reverse=True):
                try:
                    p.unlink()
                except IsADirectoryError:
                    pass
        tmp_path.mkdir(parents=True, exist_ok=True)

        repo_root = Path(__file__).resolve().parents[3]  # project root
        _write_positions_effective(tmp_path)

        script = repo_root / "nbsi" / "phase4" / "scripts" / "run_phase4.py"
        cmd = [
            sys.executable, str(script),
            "--mode", "route",
            "--dry-run", "true",
            "--from", str(tmp_path / "artifacts" / "phase3"),
            "--emit-csv", "true",
        ]
        env = os.environ.copy()
        env["NBSI_OUT_ROOT"] = str(tmp_path)
        subprocess.check_call(cmd, cwd=repo_root, env=env)

        csv_path = tmp_path / "artifacts" / "phase4" / "orders_intents.csv"
        pq_path  = tmp_path / "artifacts" / "phase4" / "orders_intents.parquet"
        qa_path  = tmp_path / "artifacts" / "phase4" / "qa_phase4.log"

        self.assertTrue(csv_path.exists(), "orders_intents.csv was not written")
        self.assertTrue(pq_path.exists(),  "orders_intents.parquet was not written")
        df_csv = pd.read_csv(csv_path)
        self.assertGreater(len(df_csv), 0, "CSV should have at least one intent row")
        self.assertTrue(qa_path.exists())
        qa_txt = qa_path.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("emitted CSV", qa_txt)


if __name__ == "__main__":
    unittest.main()


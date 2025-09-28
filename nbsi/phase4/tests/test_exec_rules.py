import unittest
import pandas as pd
import numpy as np
from nbsi.phase4.exec.simulator import ExecConfig, build_daily_positions, simulate_opg


class TestExecRules(unittest.TestCase):
    def setUp(self):
        self.cfg = ExecConfig()
        dates = pd.date_range("2025-09-01", periods=6, freq="B")
        tickers = self.cfg.universe
        rng = np.random.default_rng(42)
        scores = pd.DataFrame(rng.standard_normal((len(dates), len(tickers))), index=dates, columns=tickers)
        self.scores = scores

        opens = pd.DataFrame(100.0, index=dates, columns=tickers)
        closes = opens * (1.0 + 0.001)
        self.opens, self.closes = opens, closes
        self.sectors = {t: t for t in tickers}

    def test_caps_and_hold(self):
        pos = build_daily_positions(self.scores, self.sectors, self.cfg)
        gross = pos.abs().sum(axis=1)
        self.assertTrue((gross <= self.cfg.gross_cap + 1e-9).all())
        for s in set(self.sectors.values()):
            g = pos.loc[:, [t for t, ss in self.sectors.items() if ss == s]].abs().sum(axis=1)
            self.assertTrue((g <= self.cfg.sector_cap + 1e-9).all())

    def test_opg_shift(self):
        pos = build_daily_positions(self.scores, self.sectors, self.cfg)
        fills, pnl, eff = simulate_opg(pos, self.opens, self.closes, self.cfg)
        self.assertTrue((fills.iloc[0].abs() == 0.0).all())
        self.assertEqual(len(pnl), len(self.opens))


if __name__ == "__main__":
    unittest.main()


import unittest
import pandas as pd
import numpy as np

from nbsi.phase4.route import build_opg_intents


TICKERS = ("XLB","XLC","XLE","XLU","XLV","XLY")


def _df(index, rows):
    """Helper to build a positions_effective DataFrame with DateTimeIndex.
    rows: list of dicts mapping ticker->weight
    """
    df = pd.DataFrame(rows, index=pd.to_datetime(index))
    # ensure all tickers present as columns in stable order
    for t in TICKERS:
        if t not in df.columns:
            df[t] = 0.0
    return df[list(TICKERS)]


class TestRouter(unittest.TestCase):
    def test_first_day_diff_vs_zero(self):
        # Day 1 only
        day1 = {
            "XLB": 0.25, "XLC": 0.25, "XLE": 0.25,
            "XLU": -0.25, "XLV": -0.25, "XLY": -0.25,
        }
        positions = _df(["2025-07-08"], [day1])

        intents = build_opg_intents(
            positions_effective=positions,
            sectors={t: t for t in TICKERS},
            gross_cap=10.0,
            sector_cap=10.0,
            tif="opg",
        )

        # 6 intents: 3 buys, 3 sells
        self.assertEqual(len(intents), 6)
        self.assertEqual(set(intents["tif"]), {"opg"})
        self.assertTrue(set(intents["action"]).issubset({"buy","sell"}))

        buys = intents[intents["action"] == "buy"]
        sells = intents[intents["action"] == "sell"]
        self.assertEqual(len(buys), 3)
        self.assertEqual(len(sells), 3)

        # weight_delta sums to ~0 per day
        s = intents.groupby("date_et")["weight_delta"].sum().iloc[0]
        self.assertLessEqual(abs(s), 1e-12)

    def test_subsequent_day_diff_vs_prior_and_noop_day(self):
        # Day 1 baseline
        day1 = {
            "XLB": 0.25, "XLC": 0.25, "XLE": 0.25,
            "XLU": -0.25, "XLV": -0.25, "XLY": -0.25,
        }
        # Day 2: two longs to zero; one short reduces (less negative);
        # bump remaining long to keep net 0. We set high caps to avoid validation trips.
        day2 = {
            "XLB": 0.0, "XLC": 0.0, "XLE": 0.60,
            "XLU": -0.10, "XLV": -0.25, "XLY": -0.25,
        }
        # Day 3: identical to Day 2 (no-op)
        day3 = dict(day2)

        positions = _df(["2025-07-08","2025-07-09","2025-07-10"], [day1, day2, day3])

        intents = build_opg_intents(
            positions_effective=positions,
            sectors={t: t for t in TICKERS},
            gross_cap=10.0,
            sector_cap=10.0,
            tif="opg",
        )

        # Day 2: only changed tickers appear
        d2 = intents[intents["date_et"] == pd.Timestamp("2025-07-09")]
        changed = {"XLB","XLC","XLE","XLU"}
        self.assertEqual(set(d2["ticker"]), changed)

        # Check deltas = w_t - w_{t-1}
        expected = {
            "XLB": 0.0 - 0.25,  # sell 0.25
            "XLC": 0.0 - 0.25,  # sell 0.25
            "XLE": 0.60 - 0.25, # buy 0.35
            "XLU": -0.10 - (-0.25), # buy 0.15
        }
        got = dict(zip(d2["ticker"], d2["weight_delta"]))
        for t, wd in expected.items():
            self.assertIn(t, got)
            self.assertLessEqual(abs(got[t] - wd), 1e-12)

        # Day 3: no intents
        d3 = intents[intents["date_et"] == pd.Timestamp("2025-07-10")]
        self.assertEqual(len(d3), 0)

    def test_schema_and_invariants(self):
        day1 = {"XLB": 0.25, "XLC": 0.25, "XLE": 0.25, "XLU": -0.25, "XLV": -0.25, "XLY": -0.25}
        day2 = {"XLB": 0.0,  "XLC": 0.0,  "XLE": 0.60, "XLU": -0.10, "XLV": -0.25, "XLY": -0.25}
        positions = _df(["2025-07-08","2025-07-09"], [day1, day2])

        intents = build_opg_intents(
            positions_effective=positions,
            sectors={t: t for t in TICKERS},
            gross_cap=10.0,
            sector_cap=10.0,
            tif="opg",
        )

        # Exact schema and values
        self.assertEqual(list(intents.columns), ["date_et","ticker","action","weight_delta","tif"])
        self.assertTrue(set(intents["action"]) <= {"buy","sell"})
        self.assertEqual(intents["tif"].unique().tolist(), ["opg"])

        # Bounds and per-day neutrality (for this synthetic fixture)
        self.assertTrue((intents["weight_delta"].abs() <= 1.0 + 1e-12).all())
        per_day = intents.groupby("date_et")["weight_delta"].sum()
        self.assertTrue(all(abs(v) <= 1e-12 for v in per_day))

    def test_cap_safety_raises(self):
        # Gross breach
        positions_gross = _df(
            ["2025-07-08"],
            [{"XLB": 0.30, "XLC": 0.30, "XLE": 0.30, "XLU": -0.30, "XLV": -0.30, "XLY": -0.30}],
        )
        with self.assertRaisesRegex(ValueError, "Gross cap"):
            build_opg_intents(
                positions_effective=positions_gross,
                sectors={t: t for t in TICKERS},
                gross_cap=1.50,
                sector_cap=1.00,
                tif="opg",
            )

        # Sector breach
        positions_sector = _df(
            ["2025-07-08"],
            [{"XLB": 0.20, "XLC": 0.00, "XLE": 0.20, "XLU": 0.00, "XLV": 0.00, "XLY": 0.00}],
        )
        sectors = {"XLB": "MATS", "XLE": "MATS", "XLC": "COMM", "XLU": "UTIL", "XLV": "HC", "XLY": "DISC"}
        with self.assertRaisesRegex(ValueError, "Sector cap"):
            build_opg_intents(
                positions_effective=positions_sector,
                sectors=sectors,
                gross_cap=10.0,
                sector_cap=0.30,
                tif="opg",
            )

    def test_missing_nan_tolerance(self):
        # Single-day with NaN: ensure NaN treated as 0 (no spurious intent)
        day1 = {"XLB": np.nan, "XLC": 0.0, "XLE": 0.10, "XLU": 0.0, "XLV": 0.0, "XLY": 0.0}
        positions = _df(["2025-07-08"], [day1])

        intents = build_opg_intents(
            positions_effective=positions,
            sectors={t: t for t in TICKERS},
            gross_cap=10.0,
            sector_cap=10.0,
            tif="opg",
        )

        # Should only include XLE; XLB (NaN) should not appear
        self.assertEqual(set(intents["ticker"]), {"XLE"})
        self.assertEqual(intents.iloc[0]["action"], "buy")
        self.assertLessEqual(abs(intents.iloc[0]["weight_delta"] - 0.10), 1e-12)


if __name__ == "__main__":
    unittest.main()


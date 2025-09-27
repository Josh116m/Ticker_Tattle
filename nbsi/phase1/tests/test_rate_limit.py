import unittest
from unittest.mock import patch, MagicMock

from nbsi.phase1.data.finnhub_client import FinnhubClient, FinnhubForbidden


class FakeResp:
    def __init__(self, status_code=200, json_data=None, headers=None, text=""):
        self.status_code = status_code
        self._json = json_data or {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class FakeTime:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def time(self):
        return self.t

    def sleep(self, x):
        self.sleeps.append(x)
        self.t += x


class RateLimitTests(unittest.TestCase):
    def test_token_bucket_enforces_rate(self):
        ft = FakeTime()
        client = FinnhubClient(api_key="x", rate_limit_config={"max_per_second": 5, "bucket_capacity": 5})
        # Patch time within module
        with patch("nbsi.phase1.data.finnhub_client.time.time", ft.time), \
             patch("nbsi.phase1.data.finnhub_client.time.sleep", ft.sleep):
            # Mock session.get to always return 200 quickly
            client.session.get = MagicMock(return_value=FakeResp(200, {"ok": True}))
            # Make 15 calls; with 5/s limit, it should take at least ~2 seconds to complete bursts beyond the first bucket
            for i in range(15):
                # vary params to bypass memo cache
                client.request("/stock/profile2", params={"symbol": f"AAPL{i}"})
            # First 5 immediate, next 10 require ~2 seconds of total sleep (5/s)
            self.assertGreaterEqual(sum(ft.sleeps), 2.0 - 1e-6)

    def test_retry_after_honored(self):
        ft = FakeTime()
        client = FinnhubClient(api_key="x", rate_limit_config={"max_per_second": 100, "bucket_capacity": 100})
        with patch("nbsi.phase1.data.finnhub_client.time.time", ft.time), \
             patch("nbsi.phase1.data.finnhub_client.time.sleep", ft.sleep):
            seq = [
                FakeResp(429, headers={"Retry-After": "2"}),
                FakeResp(200, json_data={"ok": True}),
            ]
            def _get(url, params, timeout):
                return seq.pop(0)
            client.session.get = _get
            client.request("/stock/profile2", params={"symbol": "AAPL"})
            # Should have slept at least 2 seconds due to Retry-After
            self.assertGreaterEqual(sum(ft.sleeps), 2.0 - 1e-6)

    def test_403_short_circuit(self):
        client = FinnhubClient(api_key="x", rate_limit_config={"max_per_second": 100, "bucket_capacity": 100})
        calls = {"n": 0}
        def _get(url, params, timeout):
            calls["n"] += 1
            return FakeResp(403, text="Forbidden")
        client.session.get = _get
        with self.assertRaises(FinnhubForbidden):
            client.request("/etf/holdings", params={"symbol": "SPY"})
        # Ensure endpoint marked disabled; next call should immediately raise without more HTTP calls
        calls["n"] = 0
        with self.assertRaises(FinnhubForbidden):
            client.request("/etf/holdings", params={"symbol": "SPY"})
        self.assertEqual(calls["n"], 0)


if __name__ == "__main__":
    unittest.main()


import unittest
import requests
from nbsi.phase1.data.finnhub_client import FinnhubClient

class DummyResp:
    def __init__(self, status, headers=None, json_obj=None):
        self.status_code = status
        self.headers = headers or {}
        self._json = json_obj or {}
    def json(self):
        return self._json
    def raise_for_status(self):
        raise requests.HTTPError(f"{self.status_code}")

class DummySession:
    def __init__(self, seq):
        self.seq = list(seq)
    def get(self, url, params=None, timeout=None):
        return self.seq.pop(0)

class FinnhubLimiterTest(unittest.TestCase):
    def test_retry_after_backoff_and_success(self):
        # Sequence: 429 with Retry-After=1, then 200 OK
        seq = [
            DummyResp(429, headers={'Retry-After':'1'}),
            DummyResp(200, json_obj={'ok': True}),
        ]
        cli = FinnhubClient('TOKEN', rate_limit_config={'max_per_second':25,'bucket_capacity':30}, session=DummySession(seq))
        sleeps = []
        cli._sleep = lambda s: sleeps.append(s)

        out = cli.request('/stock/profile2', params={'symbol':'SPY'})
        self.assertTrue(out['ok'])
        # Ensure we honored Retry-After
        self.assertTrue(any(abs(s-1.0) < 1e-6 for s in sleeps))

if __name__ == '__main__':
    unittest.main()


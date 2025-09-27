import time
import threading
import hashlib
import random
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional

import requests


class FinnhubForbidden(Exception):
    pass


class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

    def try_consume(self, n: int = 1) -> bool:
        with self.lock:
            self._refill()
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

    def wait_time(self, n: int = 1) -> float:
        with self.lock:
            self._refill()
            if self.tokens >= n:
                return 0.0
            needed = n - self.tokens
            return max(0.0, needed / self.refill_rate)


class FinnhubClient:
    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str, rate_limit_config: Optional[Dict[str, Any]] = None, session: Optional[requests.Session] = None):
        self.api_key = api_key
        self.session = session or requests.Session()
        cfg = rate_limit_config or {}
        self.max_per_second = cfg.get("max_per_second", 25)
        bucket_capacity = cfg.get("bucket_capacity", 30)
        self.backoff_base = cfg.get("backoff_base_seconds", 0.5)
        self.backoff_factor = cfg.get("backoff_factor", 2.0)
        self.backoff_max = cfg.get("backoff_max_seconds", 60)
        self.bucket = TokenBucket(bucket_capacity, self.max_per_second)
        self.cache: Dict[str, Any] = {}
        self.disabled_endpoints = set()
        self.sec_epoch = int(time.time())
        self.sec_count = 0

    def _cache_key(self, path: str, params: Dict[str, Any]) -> str:
        p = {k: v for k, v in params.items() if k != "token"}
        s = "&".join(f"{k}={v}" for k, v in sorted(p.items()))
        return hashlib.md5(f"{path}?{s}".encode()).hexdigest()

    def _parse_retry_after(self, v: str) -> float:
        try:
            return float(v)
        except Exception:
            try:
                dt = parsedate_to_datetime(v)
                return max(0.0, (dt - datetime.now(dt.tzinfo)).total_seconds())
            except Exception:
                return 0.0

    def _sleep(self, secs: float):
        time.sleep(secs)

    def request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if path in self.disabled_endpoints:
            raise FinnhubForbidden(f"Endpoint {path} disabled due to prior 403")
        params = dict(params or {})
        params["token"] = self.api_key
        key = self._cache_key(path, params)
        if key in self.cache:
            return self.cache[key]
        url = f"{self.BASE_URL}{path}"
        attempt = 0
        backoff = self.backoff_base
        while attempt < 5:
            if not self.bucket.try_consume():
                wait = self.bucket.wait_time()
                if wait > 0:
                    print(f"[INFO] Rate limit: sleeping {wait:.2f}s for token bucket")
                    self._sleep(wait)
                    self.bucket.try_consume()
            now = int(time.time())
            if now != self.sec_epoch:
                print(f"[INFO] Finnhub call rate: {self.sec_count} req/s (sec {self.sec_epoch})")
                self.sec_epoch, self.sec_count = now, 0
            self.sec_count += 1
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    self.cache[key] = data
                    return data
                if resp.status_code == 403:
                    self.disabled_endpoints.add(path)
                    print(f"[WARN] Endpoint {path} 403 — disabling for this run")
                    raise FinnhubForbidden("403 Forbidden")
                if resp.status_code == 429:
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        wait = self._parse_retry_after(ra)
                        print(f"[WARN] 429 on {path}, Retry-After={ra}, sleep {wait:.2f}s (attempt {attempt+1})")
                    else:
                        wait = min(self.backoff_max, backoff + random.uniform(0, backoff * 0.1))
                        print(f"[WARN] 429 on {path}, sleep {wait:.2f}s (attempt {attempt+1})")
                    self._sleep(wait)
                    backoff = min(self.backoff_max, backoff * self.backoff_factor)
                    attempt += 1
                    continue
                if resp.status_code >= 500:
                    wait = min(self.backoff_max, backoff + random.uniform(0, backoff * 0.1))
                    print(f"[WARN] {resp.status_code} on {path}, sleep {wait:.2f}s (attempt {attempt+1})")
                    self._sleep(wait)
                    backoff = min(self.backoff_max, backoff * self.backoff_factor)
                    attempt += 1
                    continue
                resp.raise_for_status()
            except requests.RequestException as e:
                if attempt >= 4:
                    raise
                wait = min(self.backoff_max, backoff + random.uniform(0, backoff * 0.1))
                print(f"[WARN] Exception on {path}: {e}, sleep {wait:.2f}s (attempt {attempt+1})")
                self._sleep(wait)
                backoff = min(self.backoff_max, backoff * self.backoff_factor)
                attempt += 1
        raise requests.HTTPError(f"Failed after {attempt} attempts: {path}")

    # Convenience wrappers
    def company_profile(self, symbol: str) -> Dict[str, Any]:
        return self.request("/stock/profile2", params={"symbol": symbol})

    def earnings_calendar(self, start_date: str, end_date: str) -> Dict[str, Any]:
        return self.request("/calendar/earnings", params={"from": start_date, "to": end_date})


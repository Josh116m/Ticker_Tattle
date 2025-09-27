from typing import Dict, Any, Iterator, List, Optional
import requests

POLY_BASE = "https://api.polygon.io"


def _iter_pages(session: requests.Session, url: str, params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    while True:
        r = session.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        yield data
        nxt = data.get("next_url")
        if not nxt:
            break
        # Polygon returns absolute next_url including scheme/host
        url = nxt
        params = {}


class PolygonClient:
    def __init__(self, api_key: str, trace: bool = False, session: Optional[requests.Session] = None):
        self.api_key = api_key
        self.session = session or requests.Session()
        # Ensure auth works for pagination next_url without query apiKey
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "X-Polygon-API-Key": api_key,
        })
        self.trace = trace

    def list_news(self, from_date: str, to_date: str, tickers: Optional[List[str]] = None, page_size: int = 50) -> List[Dict[str, Any]]:
        params = {"apiKey": self.api_key, "published_utc.gte": from_date, "published_utc.lt": to_date, "limit": page_size}
        if tickers:
            params["ticker"] = ",".join(tickers)
        url = f"{POLY_BASE}/v2/reference/news"
        items: List[Dict[str, Any]] = []
        for page in _iter_pages(self.session, url, params):
            items.extend(page.get("results", []))
        return items

    def get_aggs_daily(self, ticker: str, start: str, end: str) -> List[Dict[str, Any]]:
        url = f"{POLY_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        params = {"apiKey": self.api_key, "adjusted": "true"}
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("results", [])


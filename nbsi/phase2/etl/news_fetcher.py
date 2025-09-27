import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

ART_DIR = os.path.join('artifacts','phase2')
logger = logging.getLogger(__name__)

class PolygonNewsFetcher:
    def __init__(self, api_key: str, page_limit: int = 1000):
        self.api_key = api_key
        self.session = requests.Session()
        # Keep bearer + header key on the session so next_url remains authorized
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "X-Polygon-API-Key": api_key,
            "Accept": "application/json",
            "User-Agent": "nbelastic-phase2/1.0",
        })
        self.base_url = "https://api.polygon.io/v2/reference/news"
        self.page_limit = page_limit

    def _fetch_page(self, url: str, params: Optional[Dict] = None) -> Dict:
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def iter_news(self, start_iso: str, end_iso: str) -> Iterable[Dict]:
        # Fetch ascending by time
        params = {
            "published_utc.gte": start_iso,
            "published_utc.lte": end_iso,
            "order": "asc",
            "limit": self.page_limit,
        }
        url = self.base_url
        page = 0
        while True:
            page += 1
            data = self._fetch_page(url, params=params)
            results = data.get("results", [])
            logger.info("Polygon news page %s: %s rows", page, len(results))
            for row in results:
                yield row
            next_url = data.get("next_url")
            if not next_url:
                break
            # follow absolute next_url; do not append apiKey; keep session headers
            url = next_url
            params = None

    def fetch_window(self, date_utc: datetime) -> pd.DataFrame:
        day_start = datetime(date_utc.year, date_utc.month, date_utc.day, tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1) - timedelta(seconds=1)
        rows = list(self.iter_news(day_start.isoformat(), day_end.isoformat()))
        if not rows:
            return pd.DataFrame()
        return pd.json_normalize(rows)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def run_for_range(start_date_utc: datetime, end_date_utc: datetime, api_key: str, page_limit: int = 1000) -> List[str]:
    fetcher = PolygonNewsFetcher(api_key, page_limit=page_limit)
    written = []
    cur = start_date_utc
    while cur <= end_date_utc:
        df = fetcher.fetch_window(cur)
        out = os.path.join(ART_DIR, f"news_raw_{cur.strftime('%Y%m%d')}.parquet")
        write_parquet(df, out)
        logger.info("Wrote %s with %s rows", out, len(df))
        written.append(out)
        cur += timedelta(days=1)
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    # Minimal CLI for ad-hoc runs
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--start', required=True, help='UTC start date YYYY-MM-DD')
    p.add_argument('--end', required=True, help='UTC end date YYYY-MM-DD')
    p.add_argument('--page-limit', type=int, default=1000)
    args = p.parse_args()

    # Load secrets from phase1 secrets.yaml for reuse
    secrets = pd.read_yaml if False else None  # placeholder to avoid accidental import
    # Inline minimalist loader
    import yaml
    with open(os.path.join('nbsi','phase1','configs','secrets.yaml'),'r',encoding='utf-8') as fh:
        y = yaml.safe_load(fh)
    api_key = y['polygon_api_key']

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    run_for_range(start, end, api_key, page_limit=args.page_limit)


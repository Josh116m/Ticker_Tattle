from typing import Dict, Any, List, Optional
import requests

class AlpacaClient:
    def __init__(self, key_id: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets/v2", session: Optional[requests.Session] = None):
        self.key_id = key_id
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret_key}

    def get_positions(self) -> List[Dict[str, Any]]:
        r = self.session.get(f"{self.base_url}/positions", headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    def submit_orders(self, target_weights: Dict[str, float], when: str = "next_open") -> Dict[str, Any]:
        payload = {"when": when, "targets": target_weights}
        r = self.session.post(f"{self.base_url}/orders:target_weights", json=payload, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    def get_fills(self, date: str) -> List[Dict[str, Any]]:
        # Note: Alpaca v2 uses /activities?activity_types=FILL&date=YYYY-MM-DD
        r = self.session.get(f"{self.base_url}/account/activities", params={"activity_types": "FILL", "date": date}, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()


import os, json, sys
from datetime import datetime
import pandas as pd

URL = os.environ.get("WEBHOOK_URL")  # set in user environment (never in repo)
if not URL:
    print("[notify] WEBHOOK_URL not set; skipping")
    sys.exit(0)

root = os.environ.get("NBSI_OUT_ROOT", ".")
csv_path = os.path.join(root, "artifacts", "phase5", "daily_equity.csv")

try:
    df = pd.read_csv(csv_path)
    last = df.tail(1).iloc[0]
    dt, eq = last["date_et"], float(last["equity"])
    msg = f"Phase-5: {dt} | equity={eq:.3f} | {datetime.now().isoformat(timespec='seconds')}"
except Exception as e:
    msg = f"Phase-5: unable to read equity CSV ({e})"

payload = {"text": msg}  # Slack-compatible; also fine for Teams simple webhook
try:
    import requests  # lazy import to avoid extra CI deps

    r = requests.post(
        URL,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    print(f"[notify] {r.status_code}")
except Exception as e:
    print(f"[notify] error: {e}")


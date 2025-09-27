import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
import yaml

from nbsi.phase2.etl.news_fetcher import run_for_range
from nbsi.phase2.etl.news_cleaner import clean_day
from nbsi.phase2.features.sentiment import run_finbert, write_article_sentiment
from nbsi.phase2.features.relevance import compute_relevance, write_article_relevance
from nbsi.phase2.features.panel_builder import build_panel, write_sector_panel

ART_DIR = os.path.join('artifacts','phase2')
QA_LOG = os.path.join(ART_DIR, 'qa_phase2.log')

logger = logging.getLogger(__name__)

def fail(msg: str):
    print(f"[FAIL] {msg}")
    sys.exit(2)


def load_yaml(path: str):
    with open(path,'r',encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def business_days_utc(end_utc: datetime, n_days: int) -> List[datetime]:
    # Return last n business days (Mon-Fri), inclusive of end_utc's date
    days = []
    cur = end_utc
    while len(days) < n_days:
        if cur.weekday() < 5:
            days.append(datetime(cur.year, cur.month, cur.day, tzinfo=timezone.utc))
        cur -= timedelta(days=1)
    return list(reversed(days))


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # Preconditions
    # 1) Git tag exists
    import subprocess
    tags = subprocess.check_output(['git','tag','--list'], text=True)
    if 'nbelastic-v1.2-phase0' not in tags:
        fail('Missing git tag nbelastic-v1.2-phase0')
    # 2) Secrets present
    secrets_path = os.path.join('nbsi','phase1','configs','secrets.yaml')
    if not os.path.exists(secrets_path):
        fail('Missing secrets at nbsi/phase1/configs/secrets.yaml')
    with open(secrets_path,'r',encoding='utf-8') as fh:
        secrets = yaml.safe_load(fh)
    for key in ['polygon_api_key','finnhub_api_key','alpaca']:
        if key not in secrets:
            fail(f'Missing {key} in secrets')
    # 3) Working tree clean (only allow artifacts/* changes)
    status = subprocess.check_output(['git','status','--porcelain'], text=True)
    dirty = any(not line.strip().startswith('?? artifacts') and line.strip() for line in status.splitlines())
    if dirty:
        print(status)
        fail('Working tree is dirty; commit or stash non-artifact changes before Phase-2 run')

    cfg = load_yaml(os.path.join('nbsi','phase2','configs','config.yaml'))

    polygon_api_key = secrets['polygon_api_key']

    os.makedirs(ART_DIR, exist_ok=True)

    # 1) Fetch last 60 ET trading days of news (approx via business days UTC)
    end_utc = datetime.now(timezone.utc)
    days_utc = business_days_utc(end_utc, cfg['lookback']['build_days'])

    written = run_for_range(days_utc[0], days_utc[-1], polygon_api_key, page_limit=cfg['polygon_news']['page_limit'])

    # 2) Clean per day and gather
    clean_paths = []
    clean_rows = []
    for d in days_utc:
        ymd = d.strftime('%Y%m%d')
        raw_path = os.path.join(ART_DIR, f'news_raw_{ymd}.parquet')
        if not os.path.exists(raw_path):
            continue
        df_raw = pd.read_parquet(raw_path)
        df_clean = clean_day(df_raw, cutoff_et=cfg['embargo']['cutoff_time_et'])
        out = os.path.join(ART_DIR, f'news_clean_{ymd}.parquet')
        df_clean.to_parquet(out, index=False)
        clean_paths.append(out)
        clean_rows.append(df_clean)

    df_all_clean = pd.concat(clean_rows, ignore_index=True) if clean_rows else pd.DataFrame(columns=['article_id'])

    # 3) Features: sentiment
    try:
        df_sent = run_finbert(df_all_clean, use_gpu=cfg['features']['sentiment'].get('use_gpu', True))
    except Exception as e:
        logger.warning('FinBERT failed on GPU; retrying on CPU: %s', e)
        df_sent = run_finbert(df_all_clean, use_gpu=False)
    write_article_sentiment(df_sent)

    # 4) Features: relevance
    try:
        df_rel = compute_relevance(df_all_clean, use_gpu=cfg['features']['relevance'].get('use_gpu', True))
    except Exception as e:
        logger.warning('Sentence-Transformer failed on GPU; retrying on CPU: %s', e)
        df_rel = compute_relevance(df_all_clean, use_gpu=False)
    write_article_relevance(df_rel)

    # 5) Build sector panel - pull Phase-0 SPY proxy & RVs from artifacts/phase0_rl_qa
    spy_report = os.path.join('artifacts','phase0_rl_qa','spy_proxy_report.csv')
    price_health = os.path.join('artifacts','phase0_rl_qa','price_health.csv')
    if not (os.path.exists(spy_report) and os.path.exists(price_health)):
        fail('Missing Phase-0 RL QA artifacts; run phase0_rl_qa first')

    spy = pd.read_csv(spy_report)
    price = pd.read_csv(price_health)
    # Merge to get spy_sentiment + rv columns per date_et
    spy_daily = spy[['date_et','spy_sentiment']].merge(price[['date_et','rv20','rv60']].drop_duplicates('date_et'), on='date_et', how='left')
    spy_daily['date_et'] = pd.to_datetime(spy_daily['date_et'])

    panel = build_panel(df_all_clean, df_sent, df_rel, spy_daily,
                        half_life_hours=cfg['features']['panel']['recency_half_life_hours'],
                        extreme_thr=cfg['features']['panel']['extreme_polarity_threshold'])
    write_sector_panel(panel)

    # 6) QA log
    with open(QA_LOG,'w',encoding='utf-8') as fh:
        ok = True
        def w(s=''):
            print(s)
            fh.write(str(s)+'\n')
        # Pagination QA proxy (from raw day samples)
        w('--- News pagination QA ---')
        sample_days = [p for p in written if os.path.getsize(p) > 0][:2]
        for p in sample_days:
            df = pd.read_parquet(p)
            w(f'{os.path.basename(p)} rows={len(df)}')
            if 'id' in df.columns:
                dups = int(df['id'].duplicated().sum())
                w(f'dup ids: {dups}')
                if dups > 0: ok = False
            if 'published_utc' in df.columns:
                pdt = pd.to_datetime(df['published_utc'], utc=True, errors='coerce')
                non_inc = (pdt.fillna(pd.Timestamp.max).diff().dropna() >= pd.Timedelta(0)).all()
                w(f'published_utc non-decreasing (asc fetch): {bool(non_inc)}')
                if not bool(non_inc): ok = False
        w('\n--- Finnhub limiter QA (from previous run logs) ---')
        # We reference phase0_rl_qa/run.log for samples
        run_log = os.path.join('artifacts','phase0_rl_qa','run.log')
        if os.path.exists(run_log):
            with open(run_log,'r',encoding='utf-8',errors='ignore') as lf:
                lines = [ln for ln in lf.readlines() if 'Finnhub call rate' in ln or '429 on /stock/profile2' in ln]
            for ln in lines[-10:]:
                w(ln.strip())
        else:
            w('No run.log available; skip samples')
        w('\n--- Panel schema & head/tail ---')
        req_cols = ['date_et','sector','n_articles','stale_share','mean_polarity','std_polarity','pct_extreme','conf_mean','rel_weight_sum','spy_sentiment','rv20','rv60']
        miss = [c for c in req_cols if c not in panel.columns]
        w(f'missing columns: {miss}')
        if miss: ok = False
        w(panel.head(5).to_string(index=False))
        w(panel.tail(5).to_string(index=False))
        # Date coverage check against Phase-0 14d window
        cov = pd.read_csv(os.path.join('artifacts','phase0_rl_qa','coverage_report.csv'))
        last14 = pd.to_datetime(cov['date_et']).dt.normalize().unique()[-14:]
        pnl_dates = pd.to_datetime(panel['date_et']).dt.normalize().unique()
        w(f'panel unique dates: {len(pnl_dates)}; last date: {pnl_dates[-1] if len(pnl_dates)>0 else None}')
        w(f'phase0 last14 unique dates: {len(last14)}; last date: {last14[-1] if len(last14)>0 else None}')
        ok14 = set(pd.to_datetime(last14)).issubset(set(pd.to_datetime(pnl_dates)))
        w(f'last 14 ET dates covered by panel: {ok14}')
        if not ok14: ok = False
        w('\n--- SPY proxy fallback rule note ---')
        w('If n_primary < 8 then fallback to SPY-tagged only (applied in Phase-0; panel uses provided spy_sentiment)')

    if ok:
        print('PHASE 2 COMPLETE — PANEL READY')
    else:
        print('PHASE 2 FAIL — see qa log')
        sys.exit(3)

if __name__ == '__main__':
    main()


import os, sys
import pandas as pd

art_dir = os.path.join('artifacts','phase1_smoke')
log_tests = os.path.join(art_dir,'tests.log')
log_out = os.path.join(art_dir,'final_verifications.log')
news_path = os.path.join(art_dir,'news_20250926.parquet')

p0 = os.path.join('nbsi_phase0','artifacts')
cov_csv = os.path.join(p0,'coverage_report.csv')
spy_csv = os.path.join(p0,'spy_proxy_report.csv')
price_csv = os.path.join(p0,'price_health.csv')
earn_csv = os.path.join(p0,'earnings_density.csv')

lines = []

# 1) Finnhub throttle semantics: find 429 + Retry-After lines
try:
    with open(log_tests,'r',encoding='utf-8',errors='ignore') as fh:
        tlog = fh.read().splitlines()
    ra = [ln for ln in tlog if '429 on /stock/profile2' in ln]
    rate_sleeps = [ln for ln in tlog if 'Rate limit: sleeping' in ln]
    lines.append('--- Finnhub limiter evidence ---')
    for ln in ra[:3]:
        lines.append(ln)
    for ln in rate_sleeps[:3]:
        lines.append(ln)
except Exception as e:
    lines.append(f'Finnhub log parse error: {e}')

# 2) Polygon pagination: >50 items, duplicates, order
try:
    df_news = pd.read_parquet(news_path)
    lines.append('\n--- Polygon news pagination check ---')
    lines.append(f'news rows: {len(df_news)} (expect > 50)')
    has_id = 'id' in df_news.columns
    lines.append(f'id column present: {has_id}')
    if has_id:
        dup = int(df_news['id'].duplicated().sum())
        lines.append(f'duplicate id rows: {dup}')
    if 'published_utc' in df_news.columns:
        pdt = pd.to_datetime(df_news['published_utc'], errors='coerce', utc=True)
        non_increasing = (pdt.fillna(pd.Timestamp.max).diff().dropna() <= pd.Timedelta(0)).all()
        lines.append(f'published_utc non-increasing: {bool(non_increasing)}')
    lines.append('news head 2:')
    lines.append(df_news.head(2).to_string(index=False))
    lines.append('news tail 2:')
    lines.append(df_news.tail(2).to_string(index=False))
except Exception as e:
    lines.append(f'Polygon pagination check error: {e}')

# 3) Tiny QA checklist against Phase-0 artifacts
try:
    lines.append('\n--- coverage_report.csv head/tail (5 each) ---')
    cov = pd.read_csv(cov_csv)
    lines.append(cov.head(5).to_string(index=False))
    lines.append(cov.tail(5).to_string(index=False))
    dts = pd.to_datetime(cov['date_et']).sort_values().dt.normalize().unique()
    lines.append(f'coverage dates: {str(pd.Timestamp(dts[0]).date())} -> {str(pd.Timestamp(dts[-1]).date())}; {len(dts)} unique dates')
    bad_streaks = []
    for sec, g in cov.groupby('sector'):
        g = g.sort_values('date_et')
        zero = (g['n_articles'] == 0).astype(int)
        max_streak = 0
        cur = 0
        for z in zero:
            if z:
                cur += 1
                max_streak = max(max_streak, cur)
            else:
                cur = 0
        if max_streak > 2:
            bad_streaks.append((sec, max_streak))
    lines.append(f'all-zero >2-day streaks: {bad_streaks if bad_streaks else "NONE"}')
except Exception as e:
    lines.append(f'coverage QA error: {e}')

try:
    lines.append('\n--- spy_proxy_report.csv QA ---')
    spy = pd.read_csv(spy_csv)
    fallback_true = spy[spy['used_fallback'] == True]
    lines.append(f'total days: {len(spy)}; fallback days: {len(fallback_true)}')
    viol = fallback_true[fallback_true['n_primary'] >= 8]
    lines.append(f'fallback with n_primary>=8 rows: {len(viol)}')
    lines.append(spy.head(5).to_string(index=False))
    lines.append(spy.tail(5).to_string(index=False))
except Exception as e:
    lines.append(f'spy proxy QA error: {e}')

try:
    lines.append('\n--- price_health.csv QA ---')
    price = pd.read_csv(price_csv)
    missing_bars = price[price['bar_missing_flag'] == True]
    lines.append(f'missing-bars rows: {len(missing_bars)}')
    rv20_null = price['rv20'].isna().sum()
    lines.append(f'rv20 nulls: {rv20_null}')
    lines.append(price.head(5).to_string(index=False))
    lines.append(price.tail(5).to_string(index=False))
except Exception as e:
    lines.append(f'price health QA error: {e}')

try:
    lines.append('\n--- earnings_density.csv spot-check (non-zero rows) ---')
    earn = pd.read_csv(earn_csv)
    num_cols = [c for c in earn.columns if c != 'date_et']
    def row_has_nonzero(row):
        for c in num_cols:
            try:
                if float(row[c]) > 0:
                    return True
            except Exception:
                pass
        return False
    sample = earn[earn.apply(row_has_nonzero, axis=1)].head(5)
    lines.append(sample.to_string(index=False))
except Exception as e:
    lines.append(f'earnings density QA error: {e}')

os.makedirs(art_dir, exist_ok=True)
with open(log_out,'w',encoding='utf-8') as fh:
    fh.write('\n'.join(map(str,lines)))
print('WROTE', log_out)


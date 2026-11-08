"""Rebuild venue 1m caches from official ZIPs (registered in venue_cp3_inputs.json / CP2 raw_archives.json)
and compare OHLCV on every minute present in the raw archives. Independent of repository code."""
import io, json, sys, zipfile, re
import numpy as np, pandas as pd

OBJ = sys.argv[1]          # extracted objects dir
AUD = sys.argv[2]          # audit worktree
CACHES = sys.argv[3]       # venue caches dir
COLS = ['open_time','open','high','low','close','volume','close_time','qv','n','tbv','tbqv','ignore']

def read_zip(sha):
    with zipfile.ZipFile(f'{OBJ}/{sha}') as z:
        raw = z.read(z.namelist()[0])
    hdr = 0 if raw[:9] == b'open_time' else None
    df = pd.read_csv(io.BytesIO(raw), header=hdr, dtype=str)
    df.columns = COLS[:df.shape[1]]
    t = df['open_time'].astype('int64')
    unit = 'us' if t.max() > 1e14 else 'ms'   # spot archives switched to microseconds in 2025
    df.index = pd.DatetimeIndex(pd.to_datetime(t, unit=unit, utc=True)).as_unit('ns')
    return df[['open','high','low','close','volume']].astype(float)

def compare(label, shas, cache_sha, daily_shas=()):
    frames = [read_zip(s) for s in shas]
    raw = pd.concat(frames)
    dup = raw.index.duplicated(keep='first')
    for s in daily_shas:  # daily supplements replace/extend
        d = read_zip(s); raw = pd.concat([raw[~raw.index.isin(d.index)], d])
    raw = raw[~raw.index.duplicated(keep='first')].sort_index()
    cache = pd.read_parquet(f'{CACHES}/{cache_sha}')
    common = cache.index.intersection(raw.index)
    sub_c, sub_r = cache.loc[common], raw.loc[common]
    eq = {c: int((sub_c[c].to_numpy() != sub_r[c].to_numpy()).sum()) for c in ['open','high','low','close','volume']}
    missing_in_raw = cache.index.difference(raw.index)
    flagged = cache.loc[missing_in_raw, 'missing'].all() if len(missing_in_raw) else True
    vol0 = (sub_r['volume'] == 0) & (sub_r['open'] == sub_r['high']) & (sub_r['high'] == sub_r['low']) & (sub_r['low'] == sub_r['close'])
    ph_ok = bool((cache.loc[common, 'placeholder'].to_numpy() == vol0.to_numpy()).all())
    print(f'{label}: cache rows {len(cache)}, raw rows in-range {len(common)}, duplicate raw rows {int(dup.sum())}, '
          f'value mismatches {eq}, cache minutes absent from raw {len(missing_in_raw)} (all flagged missing: {flagged}), '
          f'placeholder flag consistent: {ph_ok}, bad {int(cache.bad.sum())}')
    return cache, raw

cp3 = json.load(open(f'{AUD}/reports/trend_hold_venue_round/cp3_registration/venue_cp3_inputs.json'))
idx = json.load(open(f'{AUD}/reports/external_audit_20261108/inputs/index.json'))
cache_sha = {r['logical_path'].split('/')[-1]: r['sha256'] for r in idx['references'] if (r.get('logical_path') or '').startswith('data/processed_1m/venue_')}
def arch(asset, venue, kind='klines'):
    return [a for a in cp3['archives'] if a['asset'] == asset and a['venue'] == venue and a['kind'] == kind]
for asset in ('BTC', 'ETH'):
    for venue in ('UM', 'SPOT', 'CM'):
        a = arch(asset, venue)
        monthly = [x['sha256'] for x in a if '/monthly/' in x['url']]
        daily = [x['sha256'] for x in a if '/daily/' in x['url']]
        compare(f'CP3 {asset} {venue} ({len(monthly)} monthly, {len(daily)} daily)', monthly, cache_sha[f'venue_cp3_{asset}_{venue}_1m.parquet'], daily)
# CP2 BTC SPOT / CM
cp2 = json.load(open(f'{AUD}/reports/trend_hold_venue_round/cp2_registration/raw_archives.json'))
def walk(x, out):
    if isinstance(x, dict):
        if isinstance(x.get('url'), str) and x['url'].endswith('.zip') and 'sha256' in x: out.append(x)
        for v in x.values(): walk(v, out)
    elif isinstance(x, list):
        for v in x: walk(v, out)
lst = []; walk(cp2, lst)
sup = []; walk(json.load(open(f'{AUD}/reports/trend_hold_venue_round/cp2_registration/input_revision_2/cm_daily_supplements.json')), sup)
spot = [x['sha256'] for x in lst if '/spot/' in x['url'] and '/klines/' in x['url']]
cm = [x['sha256'] for x in lst if '/futures/cm/' in x['url'] and '/klines/' in x['url']]
compare(f'CP2 BTC SPOT ({len(spot)} monthly)', spot, cache_sha['venue_cp2_BTC_SPOT_1m_IS.parquet'])
compare(f'CP2 BTC CM ({len(cm)} monthly, {len(sup)} daily)', cm, cache_sha['venue_cp2_BTC_CM_1m_IS.parquet'], [x['sha256'] for x in sup])

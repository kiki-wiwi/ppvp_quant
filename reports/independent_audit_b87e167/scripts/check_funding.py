"""Funding lineage: caches vs canonical registry vs raw Vision ZIPs vs saved dapi pages; UM/CM plausibility."""
import io, json, sys, zipfile
from decimal import Decimal
import numpy as np, pandas as pd

OBJ, AUD, CACHES = sys.argv[1:4]
cp3 = json.load(open(f'{AUD}/reports/trend_hold_venue_round/cp3_registration/venue_cp3_inputs.json'))
idx = json.load(open(f'{AUD}/reports/external_audit_20261108/inputs/index.json'))
cache_sha = {r['logical_path'].split('/')[-1]: r['sha256'] for r in idx['references'] if (r.get('logical_path') or '').startswith('data/processed_1m/venue_')}

def vision_records(asset, venue):
    out = {}
    for a in cp3['archives']:
        if a['asset'] == asset and a['venue'] == venue and a['kind'] == 'fundingRate':
            with zipfile.ZipFile(f"{OBJ}/{a['sha256']}") as z:
                raw = z.read(z.namelist()[0])
            df = pd.read_csv(io.BytesIO(raw), dtype=str)
            for t, r in zip(df.iloc[:, 0].astype('int64'), df.iloc[:, -1]):
                out[int(round(t / 60000) * 60000)] = Decimal(r)
    return out

def dapi_records(paths):
    out = {}
    for p in paths:
        for rec in json.load(open(f'{AUD}/{p}')):
            out[int(round(int(rec['fundingTime']) / 60000) * 60000)] = Decimal(rec['fundingRate'])
    return out

for key, asset, venue in (('BTC_UM', 'BTC', 'UM'), ('ETH_UM', 'ETH', 'UM'), ('BTC_CM', 'BTC', 'CM'), ('ETH_CM', 'ETH', 'CM')):
    canon = cp3['funding'][key]['canonical_records']
    cache = pd.read_parquet(f"{CACHES}/{cache_sha[f'venue_cp3_{asset}_{venue}_funding.parquet']}")
    ct = (cache.index.asi8 // 10**6).tolist()
    cm = {r['funding_time_ms']: Decimal(r['funding_rate']) for r in canon}
    first_canon = min(cm)
    warm = [(t, v) for t, v in zip(ct, cache['funding_rate'].to_numpy()) if t < first_canon]
    body = [(t, v) for t, v in zip(ct, cache['funding_rate'].to_numpy()) if t >= first_canon]
    same_times = sorted(cm) == [t for t, _ in body]
    vals = all(float(cm[t]) == v for t, v in body)
    print(f'  {key}: warmup points in cache before canonical start: {len(warm)}')
    vis = vision_records(asset, venue)
    src = {}
    for r in canon: src.setdefault(r['source'], []).append(r)
    vmis = sum(1 for r in src.get('vision', []) if vis.get(r['funding_time_ms']) != Decimal(r['funding_rate']))
    paths = sorted({r['source_path'] for r in canon if r['source'] != 'vision'})
    dp = dapi_records(paths) if paths else {}
    dmis = sum(1 for r in canon if r['source'] != 'vision' and dp.get(r['funding_time_ms']) != Decimal(r['funding_rate']))
    # overlap where both vision and dapi exist for the same timestamp
    ov = [t for t in dp if t in vis]
    ovmis = sum(1 for t in ov if dp[t] != vis[t])
    grid_ok = all((t - ct[0]) % (8 * 3600 * 1000) == 0 for t in ct)
    print(f'{key}: cache {len(cache)} == canon {len(canon)} times-equal {same_times} values-equal {vals}; sources {{{", ".join(f"{k}:{len(v)}" for k,v in src.items())}}}; '
          f'vision mismatches {vmis}; dapi-sourced mismatches vs saved pages {dmis}; vision/dapi overlap {len(ov)} mismatches {ovmis}; 8h grid {grid_ok}; '
          f'first {pd.Timestamp(ct[0], unit="ms", tz="UTC")} last {pd.Timestamp(ct[-1], unit="ms", tz="UTC")}')

# plausibility: CM vs UM funding by source segment
for asset in ('BTC', 'ETH'):
    um = pd.read_parquet(f"{CACHES}/{cache_sha[f'venue_cp3_{asset}_UM_funding.parquet']}")['funding_rate']
    canon = cp3['funding'][f'{asset}_CM']['canonical_records']
    cmr = pd.Series({pd.Timestamp(r['funding_time_ms'], unit='ms', tz='UTC'): float(r['funding_rate']) for r in canon})
    srcs = pd.Series({pd.Timestamp(r['funding_time_ms'], unit='ms', tz='UTC'): r['source'] for r in canon})
    j = pd.concat([um.rename('um'), cmr.rename('cm'), srcs.rename('src')], axis=1, join='inner')
    for s, g in j.groupby('src'):
        print(f'{asset} CM vs UM [{s}] n={len(g)} corr={g.um.corr(g.cm):.3f} mean UM={g.um.mean():.6f} mean CM={g.cm.mean():.6f} median |diff|={np.median(np.abs(g.um-g.cm)):.6f}')
# CP2 BTC CM funding cache vs CP3 canonical & CP2 pages, overlap window
cp2c = pd.read_parquet(f"{CACHES}/{cache_sha['venue_cp2_BTC_CM_funding_IS.parquet']}")['funding_rate']
pages = sorted(__import__('glob').glob(f'{AUD}/reports/trend_hold_venue_round/cp2_registration/funding_api_pages/*.json'))
dp = {}
for p in pages:
    d = json.load(open(p)); recs = d if isinstance(d, list) else d.get('response', d.get('records', []))
    for rec in recs:
        dp[int(round(int(rec['fundingTime']) / 60000) * 60000)] = float(rec['fundingRate'])
hit = sum(1 for t, v in zip(cp2c.index.asi8 // 10**6, cp2c.to_numpy()) if dp.get(int(t)) == v)
print(f'CP2 BTC CM funding cache {len(cp2c)} records; equal to saved CP2 dapi page value at same time: {hit}; pages {len(pages)} records {len(dp)}')
um5 = pd.read_parquet(f'{AUD}/data/processed/BTCUSDT_funding.parquet')['funding_rate']; um5.index = um5.index.round('min')
j = pd.concat([um5.rename('um'), cp2c.rename('cm')], axis=1, join='inner')
early = j[j.index < pd.Timestamp('2022-07-01', tz='UTC')]; late = j[j.index >= pd.Timestamp('2022-07-01', tz='UTC')]
print(f'CP2 BTC CM vs UM early(dapi) n={len(early)} corr={early.um.corr(early.cm):.3f}; late(vision) n={len(late)} corr={late.um.corr(late.cm):.3f}; mean CM early {early.cm.mean():.6f} late {late.cm.mean():.6f}')

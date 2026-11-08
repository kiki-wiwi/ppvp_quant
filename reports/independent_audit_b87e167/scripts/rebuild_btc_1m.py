"""Independent rebuild of BTCUSDT UM 1m IS from official monthly ZIPs; compare with cache."""
import io, json, sys, zipfile, hashlib
import numpy as np, pandas as pd
inp = sys.argv[1]; manifest = json.load(open(sys.argv[2]))
COLS = ['open_time','open','high','low','close','volume','close_time','quote_volume','count','taker_buy_volume','taker_buy_quote_volume','ignore']
def read_zip(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist(); assert len(names) == 1, names
        raw = z.read(names[0])
    first = raw.split(b'\n', 1)[0]
    header = 0 if first.startswith(b'open_time') else None
    df = pd.read_csv(io.BytesIO(raw), header=header, names=None if header == 0 else COLS, dtype=str)
    df.columns = COLS
    return df
frames = []
for f in manifest['files']['perp_1m']:
    d = read_zip(f"{inp}/{f['sha256']}"); d['src'] = 0
    assert len(d) == f['rows'], (f['url'], len(d), f['rows'])
    frames.append(d)
raw = pd.concat(frames, ignore_index=True)
ot = pd.to_datetime(raw['open_time'].astype('int64'), unit='ms', utc=True)
assert not ot.duplicated().any()
daily = manifest['adjudicated_days'][0]
dd = read_zip(f"{inp}/{daily['daily_1m_archive']['sha256']}"); dd['src'] = 1
dot = pd.to_datetime(dd['open_time'].astype('int64'), unit='ms', utc=True)
keep = ~ot.isin(dot)
mine = pd.concat([raw[keep.to_numpy()], dd], ignore_index=True)
mine.index = pd.to_datetime(mine['open_time'].astype('int64'), unit='ms', utc=True).dt.as_unit('ns') if hasattr(pd.Series,'dt') else None
mine.index = pd.DatetimeIndex(pd.to_datetime(mine['open_time'].astype('int64'), unit='ms', utc=True)).as_unit('ns')
mine = mine.sort_index()
cache = pd.read_parquet(f"{inp}/{sys.argv[3]}")
print('rows mine', len(mine), 'cache', len(cache), 'index equal', mine.index.equals(cache.index))
for c in ['open','high','low','close','volume','quote_volume','count','taker_buy_volume']:
    a = mine[c].astype(np.float64).to_numpy(); b = cache[c].to_numpy()
    print(f'{c:18s} exact_equal={np.array_equal(a,b)} maxabs={np.nanmax(np.abs(a-b)):.3g}')
print('raw_close_time equal', np.array_equal(mine['close_time'].astype('int64').to_numpy(), cache['raw_close_time_ms'].to_numpy()))
print('src equal', np.array_equal(mine['src'].to_numpy(), cache['src'].to_numpy()))
# monthly vs daily difference on 2023-11-10
mon = raw[(~keep).to_numpy()].copy(); mon.index = dot.values if False else pd.DatetimeIndex(pd.to_datetime(mon['open_time'].astype('int64'),unit='ms',utc=True))
dd.index = pd.DatetimeIndex(dot)
diff = {c: int((mon[c].astype(float).values != dd[c].astype(float).values).sum()) for c in ['open','high','low','close','volume']}
print('2023-11-10 monthly vs daily differing minutes per column', diff)
# grid flags
grid = pd.date_range('2020-01-01', '2024-01-01', freq='1min', tz='UTC', inclusive='left').as_unit('ns')
m2 = mine.reindex(grid)
o,h,l,c,v = [m2[k].astype(float) for k in ['open','high','low','close','volume']]
missing = c.isna(); placeholder = (v == 0) & (o == h) & (h == l) & (l == c)
print('grid', len(grid), 'missing', int(missing.sum()), 'zero-vol', int((v==0).sum()), 'placeholder(bad)', int(placeholder.sum()))
bad_days = pd.Series(placeholder.values | missing.values, index=grid).groupby(grid.floor('D')).sum()
print('days with bad minutes', int((bad_days>0).sum()), bad_days[bad_days>0].to_dict())
# content hash per repository definition (independent re-implementation of documented rule)
h = hashlib.sha256(); h.update(np.asarray(cache.index.asi8, dtype='<i8').tobytes())
for col in sorted(cache.columns):
    a = cache[col].to_numpy(); a = a.astype('<f8') if a.dtype.kind=='f' else a.astype('<i8') if a.dtype.kind in 'iu' else a.astype('u1')
    h.update(col.encode()); h.update(np.ascontiguousarray(a).tobytes())
print('cache content sha256', h.hexdigest())

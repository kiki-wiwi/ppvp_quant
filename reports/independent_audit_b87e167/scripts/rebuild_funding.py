import io, json, sys, zipfile
import numpy as np, pandas as pd
inp, man, pq, ds = sys.argv[1:5]
m = json.load(open(man))['datasets'][ds]
rows = []
for f in m['files']:
    with zipfile.ZipFile(f"{inp}/{f['sha256']}") as z:
        raw = z.read(z.namelist()[0])
    first = raw.split(b'\n',1)[0]
    df = pd.read_csv(io.BytesIO(raw), header=0 if first[:1].isalpha() else None)
    if df.shape[1] == 3 and not first[:1].isalpha(): df.columns = ['calc_time','funding_interval_hours','last_funding_rate']
    assert len(df) == f['rows'], (f['url'], len(df), f['rows'])
    rows.append(df)
raw = pd.concat(rows, ignore_index=True)
print('raw cols', list(raw.columns), 'rows', len(raw))
t = pd.to_datetime(raw['calc_time'].astype('int64'), unit='ms', utc=True)
cache = pd.read_parquet(pq)
print('cache cols', list(cache.columns), cache.index.name, len(cache))
print('cache index equals raw calc_time:', pd.DatetimeIndex(t).as_unit('ns').equals(cache.index.as_unit('ns')))
print('rate exact equal:', np.array_equal(raw['last_funding_rate'].astype(float).to_numpy(), cache['funding_rate'].to_numpy()))
print('interval hours unique', raw['funding_interval_hours'].unique())
off = (t - t.dt.round('min')).dt.total_seconds()*1000
print('max abs ms offset from minute', off.abs().max(), 'n nonzero', int((off!=0).sum()))
r = t.dt.round('min')
IS = (r >= pd.Timestamp('2020-01-01',tz='UTC')) & (r < pd.Timestamp('2024-01-01',tz='UTC'))
grid = pd.date_range('2020-01-01','2024-01-01',freq='8h',tz='UTC',inclusive='left')
print('IS settlements', int(IS.sum()), 'grid', len(grid), 'equal', pd.DatetimeIndex(r[IS]).equals(grid))
x = raw['last_funding_rate'].astype(float)[IS.to_numpy()]
print('IS funding mean %.6g, frac positive %.3f, sum %.5f, max %.5f, min %.5f' % (x.mean(), (x>0).mean(), x.sum(), x.max(), x.min()))

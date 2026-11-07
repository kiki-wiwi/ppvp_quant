"""BTC-IS-only official archive adapter; no downloads, repair, or future reads.

Pure parsers accept supplied bytes for synthetic validation. The real loading
entry checks the two physical stage switches before touching the filesystem.
The root runner separately verifies the frozen source and archive registration.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile
import numpy as np
import pandas as pd
from . import scope

ROOT=Path(__file__).resolve().parents[1]
MANIFEST='reports/trend_hold_venue_round/cp2_registration/raw_archives.json'
VISION_BASE='https://data.binance.vision/data'
IS_START=pd.Timestamp('2020-01-01',tz='UTC')
IS_END=pd.Timestamp('2024-01-01',tz='UTC')
CM_EARLIEST=pd.Timestamp('2020-08-11',tz='UTC')
CM_COMPLETE_START=pd.Timestamp('2020-08-12',tz='UTC')
CM_COMMON_START=pd.Timestamp('2020-09-01',tz='UTC')
CM_FUNDING_MANIFEST='reports/trend_hold_venue_round/cp2_registration/cm_funding.json'
MINUTE=pd.Timedelta(minutes=1)
CACHE_PATHS={
    'SPOT':'data/processed_1m/venue_cp2_BTC_SPOT_1m_IS.parquet',
    'CM':'data/processed_1m/venue_cp2_BTC_CM_1m_IS.parquet',
    'CM_funding':'data/processed_1m/venue_cp2_BTC_CM_funding_IS.parquet',
}
OHLCV=('open','high','low','close','volume')


def require_cp2():
    if not scope.BTC_IS_UNLOCKED or scope.POST_IS_AND_ETH_UNLOCKED:
        raise scope.RealDataLockedError('CP2 requires BTC IS unlocked and POST-IS/ETH still locked')


def _month(month,venue):
    if venue not in ('SPOT','CM') or not re.fullmatch(r'20\d{2}-\d{2}',str(month)):
        raise ValueError('unregistered BTC archive month/venue')
    start=pd.Timestamp(month+'-01',tz='UTC')
    end=start+pd.offsets.MonthBegin(1)
    lower=IS_START if venue=='SPOT' else pd.Timestamp('2020-08-01',tz='UTC')
    if not lower<=start<end<=IS_END:raise ValueError('archive month outside BTC IS')
    return start,end


def _table(raw,columns):
    if not isinstance(raw,bytes) or not raw.strip():raise ValueError('nonempty supplied CSV bytes required')
    first=raw.splitlines()[0].split(b',')[0].strip().lstrip(b'\xef\xbb\xbf')
    header=not first.isdigit()
    table=pd.read_csv(io.BytesIO(raw),header=0 if header else None)
    if table.empty or table.shape[1]!=columns:raise ValueError('unexpected CSV column count or empty archive')
    return table


def _timestamps(values,label):
    numeric=pd.to_numeric(values,errors='raise').to_numpy()
    if not np.isfinite(numeric).all() or np.any(numeric!=np.floor(numeric)):
        raise ValueError(label+': finite integer timestamps required')
    micro=numeric>10**14
    if micro.any() and not micro.all():raise ValueError(label+': mixed timestamp units')
    unit='us' if micro.all() else 'ms'
    dates=pd.DatetimeIndex(pd.to_datetime(numeric.astype(np.int64),unit=unit,utc=True)).as_unit('ns')
    if dates.hasnans:raise ValueError(label+': nonfinite dates')
    return dates


def _ordered(index,label):
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(label+': unique ordered timestamps required')


def _bounds(index,start,end,label):
    if len(index)==0 or index[0]<start or index[-1]>=end:
        raise ValueError(label+': timestamp outside requested month')


def parse_klines(raw,month,venue):
    """Parse one native BTC monthly 1m archive (ms/us; header/no header)."""
    start,end=_month(month,venue)
    table=_table(raw,12)
    index=_timestamps(table.iloc[:,0],'kline open')
    _ordered(index,'klines');_bounds(index,start,end,'klines')
    if np.any(index.asi8%scope.MINUTE_NS):raise ValueError('kline open must lie on minute boundary')
    values=table.iloc[:,1:6].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():raise ValueError('finite native OHLCV required')
    if np.any(values[:,:4]<=0) or np.any(values[:,4]<0):raise ValueError('positive native prices and nonnegative volume required')
    op,high,low,close,_=values.T
    if np.any(high<np.maximum(op,close)) or np.any(low>np.minimum(op,close)):
        raise ValueError('invalid native OHLC bounds')
    raw_close=_timestamps(table.iloc[:,6],'kline close')
    # Official close-time precision may be ms or us; both terminate within the last ms.
    close_delta=raw_close.asi8-index.asi8
    if np.any((close_delta<scope.MINUTE_NS-1_000_000)|(close_delta>=scope.MINUTE_NS)):
        raise ValueError('native close time does not end its one-minute bar')
    result=pd.DataFrame(values,index=index,columns=OHLCV)
    result.index.name='open_time'
    return result


def parse_funding(raw,month):
    """Keep actual CM settlement records, rounding official timestamp jitter once."""
    start,end=_month(month,'CM')
    table=_table(raw,3)
    original=_timestamps(table.iloc[:,0],'funding')
    _ordered(original,'raw funding')
    rounded=original.round('min').as_unit('ns')
    _ordered(rounded,'rounded funding');_bounds(rounded,start,end,'funding')
    rates=pd.to_numeric(table.iloc[:,2],errors='raise').to_numpy(dtype=np.float64)
    intervals=pd.to_numeric(table.iloc[:,1],errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(rates).all() or not np.isfinite(intervals).all() or np.any(intervals<=0):
        raise ValueError('finite funding rates and positive intervals required')
    rounded.name='funding_time'
    return pd.Series(rates,index=rounded,name='funding_rate')


def build_grid(frame,start,end):
    """R5 missing/flat-zero-volume bad rule; no backward fill or valid fabricated bar."""
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    if start.tz is None or end.tz is None:raise ValueError('aware grid bounds required')
    start,end=start.tz_convert('UTC').as_unit('ns'),end.tz_convert('UTC').as_unit('ns')
    index=scope.utc_index(frame.index)
    if not len(index) or start>=end or np.any(index.asi8%scope.MINUTE_NS):raise ValueError('invalid native minute grid')
    if index[0]<start or index[-1]>=end:raise ValueError('native frame outside requested grid')
    expected=pd.date_range(start,end,inclusive='left',freq='min').as_unit('ns')
    result=frame.loc[:,list(OHLCV)].copy().reindex(expected)
    result.index.name='open_time'
    missing=~result.index.isin(index)
    zero=result['volume'].to_numpy()==0
    flat=(result['open']==result['high'])&(result['high']==result['low'])&(result['low']==result['close'])
    placeholder=zero&flat.to_numpy()
    bad=missing|placeholder
    result['missing']=missing;result['placeholder']=placeholder
    result['bad']=bad;result['archive_gap']=missing
    prior=result['close'].where(~bad).ffill()
    if prior.isna().any():raise ValueError('initial valid native close missing; backward fill forbidden')
    for column in OHLCV[:4]:result.loc[bad,column]=prior.loc[bad]
    result.loc[bad,'volume']=0.
    a=result.index.searchsorted(start.ceil('D'))
    b=result.index.searchsorted(end.floor('D'))
    if b>a and result['bad'].iloc[a:b].to_numpy().reshape(-1,1440).all(axis=1).any():
        raise ValueError('complete UTC day has no effective minute')
    if not np.isfinite(result.loc[:,list(OHLCV)].to_numpy()).all():raise ValueError('nonfinite completed grid')
    result.attrs=dict(frame.attrs)
    return result


def cm_native_start(frame):
    flat=(frame['open']==frame['high'])&(frame['high']==frame['low'])&(frame['low']==frame['close'])
    valid=~((frame['volume']==0)&flat)
    if not valid.any():raise ValueError('CM has no effective native minute')
    start=scope.utc_index(frame.index[valid])[0]
    if not CM_EARLIEST<=start<=CM_COMPLETE_START:
        raise ValueError('CM first native minute must be >= August 11 and cover August 12 complete day')
    return start


def unzip_csv(blob,filename):
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names=archive.namelist()
        expected=filename.removesuffix('.zip')+'.csv'
        if names!=[expected] or archive.getinfo(names[0]).is_dir():
            raise ValueError('official zip must contain a single expected CSV')
        return archive.read(names[0])


def expected_archives():
    entries=[]
    for venue,first in (('SPOT','2020-01'),('CM','2020-08')):
        symbol='BTCUSDT' if venue=='SPOT' else 'BTCUSD_PERP'
        prefix='spot' if venue=='SPOT' else 'futures/cm'
        months=pd.period_range(first,'2023-12',freq='M').astype(str)
        for month in months:
            rel=f'{prefix}/monthly/klines/{symbol}/1m/{symbol}-1m-{month}.zip'
            entries.append((venue,'klines',month,rel))
            if venue=='CM' and month>='2022-07':
                rel=f'{prefix}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip'
                entries.append((venue,'fundingRate',month,rel))
    return tuple(entries)


def validate_archive_manifest(manifest):
    entries=manifest.get('archives')
    if not isinstance(entries,list):raise ValueError('archive manifest list required')
    actual=[]
    for entry in entries:
        keys=('venue','kind','month','relative_path','path','url','sha256','checksum_path','checksum_sha256')
        if not isinstance(entry,dict) or not all(k in entry for k in keys):raise ValueError('incomplete archive registration')
        rel=entry['relative_path']
        if (PurePosixPath(rel).is_absolute() or '..' in PurePosixPath(rel).parts
                or entry['path']!='data/raw/'+rel
                or entry['checksum_path']!='data/raw/'+rel+'.CHECKSUM'
                or entry['url']!=VISION_BASE+'/'+rel):
            raise ValueError('archive path/url differs from official registered path')
        if any(not re.fullmatch('[0-9a-f]{64}',entry[k]) for k in ('sha256','checksum_sha256')):
            raise ValueError('archive SHA256 must be lowercase hexadecimal')
        actual.append((entry['venue'],entry['kind'],entry['month'],rel))
    if len(actual)!=len(set(actual)) or set(actual)!=set(expected_archives()):
        raise ValueError('manifest must contain exact 107 allowed BTC IS archives')
    return sorted(entries,key=lambda e:(e['venue'],e['kind'],e['month']))


def _digest_bytes(raw):return hashlib.sha256(raw).hexdigest()


def _read_archive(root,entry):
    blob=(root/entry['path']).read_bytes()
    if _digest_bytes(blob)!=entry['sha256']:raise ValueError('archive hash mismatch: '+entry['path'])
    checksum=(root/entry['checksum_path']).read_bytes()
    if _digest_bytes(checksum)!=entry['checksum_sha256']:raise ValueError('official checksum hash mismatch')
    parts=checksum.decode('ascii').strip().split()
    filename=PurePosixPath(entry['relative_path']).name
    if len(parts)!=2 or parts[0].lower()!=entry['sha256'] or parts[1].lstrip('*')!=filename:
        raise ValueError('official checksum does not bind the registered archive')
    return unzip_csv(blob,filename)


def frame_content_sha256(frame):
    """Portable ordered UTC index/column/value digest for new input caches."""
    digest=hashlib.sha256()
    digest.update(np.asarray(frame.index.asi8,dtype='<i8').tobytes())
    for name in frame.columns:
        digest.update(name.encode()+b'\0')
        values=frame[name].to_numpy()
        dtype='u1' if pd.api.types.is_bool_dtype(values.dtype) else '<f8'
        digest.update(np.asarray(values,dtype=dtype).tobytes())
    return digest.hexdigest()


def _cache(root,path,frame):
    dest=root/path
    if dest.exists():raise FileExistsError('new CP2 input cache already exists; no automatic overwrite: '+path)
    dest.parent.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(dest)
    return {'path':path,'rows':len(frame),'content_sha256':frame_content_sha256(frame),
            'parquet_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}


def load_inputs(root=ROOT):
    """Only after the single BTC-IS unlock: validate and materialize all three venues."""
    require_cp2()  # Before importing the old real loader or touching any path.
    root=Path(root)
    from trend_hold_study.data import load_real_context,verify_environment
    environment=verify_environment()
    manifest=json.loads((root/MANIFEST).read_text())
    entries=validate_archive_manifest(manifest)
    for path in CACHE_PATHS.values():
        if (root/path).exists():raise FileExistsError('CP2 cache exists before the one-shot load: '+path)
    old,um_metadata=load_real_context(root)
    um=scope.Context(old.index,old.O,old.H,old.L,old.C,old.V,old.bad,old.archive_gap,
                     old.funding_rate,source='real',asset='BTC',venue='UM')
    del old
    parts={('SPOT','klines'):[],('CM','klines'):[],('CM','fundingRate'):[]}
    raw_rows=[]
    for entry in entries:
        raw=_read_archive(root,entry)
        frame=parse_klines(raw,entry['month'],entry['venue']) if entry['kind']=='klines' else parse_funding(raw,entry['month'])
        parts[(entry['venue'],entry['kind'])].append(frame)
        raw_rows.append({'path':entry['path'],'sha256':entry['sha256'],'parsed_rows':len(frame),
                         'first_time':frame.index[0].isoformat(),'last_time':frame.index[-1].isoformat()})
    spot_raw=pd.concat(parts[('SPOT','klines')]);cm_raw=pd.concat(parts[('CM','klines')])
    funding=load_cm_funding(root,cm_native_start(cm_raw))
    for obj,label in ((spot_raw,'Spot native'),(cm_raw,'CM native'),(funding,'CM funding')):_ordered(obj.index,label)
    native_start=cm_native_start(cm_raw)
    if funding.index[0]<native_start or funding.index[-1]>=IS_END:
        raise ValueError('CM funding outside native IS minute grid')
    spot_grid=build_grid(spot_raw,IS_START,IS_END)
    cm_grid=build_grid(cm_raw.loc[native_start:],native_start,IS_END)
    for grid in (spot_grid,cm_grid):grid.attrs['source']='real'
    spot=scope.make_context(spot_grid,source='real',asset='BTC',venue='SPOT')
    cm=scope.make_context(cm_grid,funding,source='real',asset='BTC',venue='CM')
    # Cache only after every native input and immutable Context has passed validation.
    caches={'SPOT':_cache(root,CACHE_PATHS['SPOT'],spot_grid),
            'CM':_cache(root,CACHE_PATHS['CM'],cm_grid),
            'CM_funding':_cache(root,CACHE_PATHS['CM_funding'],funding.to_frame())}
    metadata={'scope':'BTC only, all input timestamps before 2024-01-01 UTC',
              'environment':environment,'UM':um_metadata,'archives':raw_rows,'caches':caches,
              'CM_native_start':native_start.isoformat(),'CM_first_full_day':CM_COMPLETE_START.isoformat(),
              'CM_funding_registration':{'path':CM_FUNDING_MANIFEST,
                  'sha256':_digest_bytes((root/CM_FUNDING_MANIFEST).read_bytes()),
                  'source':'registered dapi and Vision canonical splice',
                  'IS_Common_points':3651,'August_native_points':int((funding.index<CM_COMMON_START).sum()),
                  'first_settlement':funding.index[0].isoformat(),'last_settlement':funding.index[-1].isoformat()},
              'new_grid_rule':'bad = missing OR flat zero-volume placeholder; archive_gap = missing; positive prior-only fill',
              'counts':{name:{'minutes':len(ctx),'bad':int(ctx.bad.sum()),
                    'archive_gap':int(ctx.archive_gap.sum()),'funding_events':int(np.isfinite(ctx.funding_rate).sum())}
                    for name,ctx in (('UM',um),('SPOT',spot),('CM',cm))}}
    return {'UM':um,'SPOT':spot,'CM':cm},metadata


def validate_cm_funding(funding,native_start):
    """Validate the approved canonical splice without synthesizing launch-month rows."""
    if not isinstance(funding,pd.Series) or not len(funding):
        raise ValueError('nonempty canonical CM funding Series required')
    index=scope.utc_index(funding.index,'canonical CM funding')
    native_start=pd.Timestamp(native_start)
    if native_start.tz is None:
        raise ValueError('aware CM native start required')
    native_start=native_start.tz_convert('UTC').as_unit('ns')
    if not CM_EARLIEST<=native_start<=CM_COMPLETE_START:
        raise ValueError('CM native start outside the registered launch boundary')
    if index[0]<native_start or index[-1]>=IS_END:
        raise ValueError('canonical CM funding outside native IS bounds')
    if np.any(index.asi8%pd.Timedelta(hours=8).value):
        raise ValueError('canonical CM funding must lie on UTC 8-hour boundaries')
    rates=funding.to_numpy(dtype=np.float64)
    if not np.isfinite(rates).all():
        raise ValueError('canonical CM funding rates must be finite')
    common=index[index>=CM_COMMON_START]
    expected=pd.date_range(CM_COMMON_START,IS_END,freq='8h',inclusive='left').as_unit('ns')
    if len(common)!=3651 or not common.equals(expected):
        raise ValueError('canonical CM IS-Common funding must contain exactly 3651 ordered settlement points')
    return pd.Series(rates,index=index,name='funding_rate')


def load_cm_funding(root,native_start):
    """Verify pinned API/Vision payloads and canonical records before using the splice."""
    require_cp2()
    from .funding_sources import verify_registered
    verified=verify_registered(Path(root))
    return validate_cm_funding(verified,native_start)

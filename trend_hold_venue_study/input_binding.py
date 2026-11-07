"""Exact approved CM daily supplements and current input identity binding.

These input-only helpers never construct features, signals, accounts or caches.
The formal caller must pass the independent BTC-IS gate before invoking file IO;
the explicitly authorized locked-stage preflight may call the same input helpers.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import pandas as pd

REG='reports/trend_hold_venue_round/cp2_registration/input_revision_2'
DAILY_MANIFEST=REG+'/cm_daily_supplements.json'
BINDING=REG+'/input_binding.json'
HISTORICAL_MANIFEST='data/manifest_impulse_1m.json'
UM_PATH='data/processed_1m/BTCUSDT_perp_1m_IS.parquet'
DAYS=('2023-08-28','2023-08-29','2023-08-30','2023-08-31')
VISION='https://data.binance.vision/data/'


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _is_sha(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _validate_entry(entry):
    if not isinstance(entry,dict) or entry.get('day') not in DAYS:
        raise ValueError('daily supplement outside the exact four approved CM dates')
    day=entry['day']
    relative=f'futures/cm/daily/klines/BTCUSD_PERP/1m/BTCUSD_PERP-1m-{day}.zip'
    if (entry.get('venue')!='CM' or entry.get('kind')!='klines'
            or entry.get('path')!='data/raw/'+relative
            or entry.get('checksum_path')!='data/raw/'+relative+'.CHECKSUM'
            or entry.get('url')!=VISION+relative):
        raise ValueError('daily supplement path or identity differs from official approved CM input')
    if type(entry.get('rows')) is not int or entry['rows']!=1440:
        raise ValueError('each daily supplement must register exactly 1440 rows')
    if not all(_is_sha(entry.get(key)) for key in ('sha256','checksum_sha256','content_sha256')):
        raise ValueError('daily supplement requires raw, checksum and content SHA256')
    return relative


def validate_daily_manifest(manifest):
    entries=manifest.get('archives') if isinstance(manifest,dict) else None
    if not isinstance(entries,list) or len(entries)!=4:
        raise ValueError('supplements must contain the exact four approved CM days')
    for entry in entries:_validate_entry(entry)
    if {entry['day'] for entry in entries}!=set(DAYS):
        raise ValueError('supplements must contain the exact four approved CM days without duplicates')
    return sorted(entries,key=lambda entry:entry['day'])


def _day_index(day):
    return pd.date_range(day,periods=1440,freq='min',tz='UTC').as_unit('ns')


def parse_daily_payload(entry,blob,checksum):
    """Authenticate supplied raw bytes before parsing exactly one complete UTC day."""
    from .data import parse_klines,unzip_csv,frame_content_sha256
    relative=_validate_entry(entry)
    if _sha(blob)!=entry['sha256']:
        raise ValueError('daily archive hash mismatch: '+entry['path'])
    if _sha(checksum)!=entry['checksum_sha256']:
        raise ValueError('daily official checksum hash mismatch')
    pieces=checksum.decode('ascii').strip().split()
    name=PurePosixPath(relative).name
    if len(pieces)!=2 or pieces[0].lower()!=entry['sha256'] or pieces[1].lstrip('*')!=name:
        raise ValueError('daily official checksum does not bind the registered archive')
    frame=parse_klines(unzip_csv(blob,name),'2023-08','CM')
    if len(frame)!=1440 or not frame.index.equals(_day_index(entry['day'])):
        raise ValueError('daily supplement must cover exactly its 1440 UTC minutes')
    if frame_content_sha256(frame)!=entry['content_sha256']:
        raise ValueError('daily parsed content hash mismatch: '+entry['day'])
    return frame


def load_daily_supplements(root):
    """Input-only registered-file verification for the four specific approved days."""
    root=Path(root)
    entries=validate_daily_manifest(json.loads((root/DAILY_MANIFEST).read_text()))
    frames=[]
    records=[]
    for entry in entries:
        frame=parse_daily_payload(entry,(root/entry['path']).read_bytes(),
                                  (root/entry['checksum_path']).read_bytes())
        frames.append(frame)
        records.append({**entry,'parsed_rows':len(frame),
                        'first_time':frame.index[0].isoformat(),'last_time':frame.index[-1].isoformat()})
    return pd.concat(frames),records


def merge_cm_supplements(monthly,daily):
    """Add only the registered absent minutes; no replacement of monthly native bars."""
    from . import scope
    expected=pd.date_range('2023-08-28','2023-09-01',freq='min',inclusive='left',tz='UTC').as_unit('ns')
    daily_index=scope.utc_index(daily.index,'daily supplement')
    if len(daily)!=5760 or not daily_index.equals(expected):
        raise ValueError('CM supplement must contain exactly four complete approved UTC days')
    monthly_index=scope.utc_index(monthly.index,'monthly native')
    if len(monthly_index.intersection(daily_index)):
        raise ValueError('CM daily supplement overlaps existing monthly native minute')
    if list(monthly.columns)!=list(daily.columns):
        raise ValueError('CM daily and monthly native columns differ')
    merged=pd.concat([monthly,daily]).sort_index()
    scope.utc_index(merged.index,'combined CM native')
    return merged


def validate_binding(document):
    if not isinstance(document,dict) or not all(key in document for key in ('UM','SPOT','CM','CM_funding')):
        raise ValueError('input binding requires all four registered input identities')
    if not isinstance(document['UM'],dict) or document['UM'].get('path')!=UM_PATH:
        raise ValueError('input binding UM path differs from the frozen BTC IS cache')
    fields={'UM':('current_parquet_sha256','content_sha256','historical_manifest_sha256'),
            'SPOT':('grid_content_sha256',),'CM':('grid_content_sha256',),
            'CM_funding':('content_sha256',)}
    for key,names in fields.items():
        if not isinstance(document[key],dict) or not all(_is_sha(document[key].get(name)) for name in names):
            raise ValueError('input binding requires valid SHA256 for '+key)
    return document


def load_binding(root):
    return validate_binding(json.loads((Path(root)/BINDING).read_text()))


def verify_um_payload(document,parquet_bytes,historical_manifest_bytes):
    """Bind current parquet bytes to the unchanged historic semantic-content hash.

    The frozen R5 loader still separately reads the bounded parquet and recomputes
    its content hash. This check adds byte identity without rewriting that manifest.
    """
    record=validate_binding(document)['UM']
    if _sha(parquet_bytes)!=record['current_parquet_sha256']:
        raise ValueError('current UM parquet bytes differ from input binding')
    if _sha(historical_manifest_bytes)!=record['historical_manifest_sha256']:
        raise ValueError('historical manifest bytes differ from input binding')
    historic=json.loads(historical_manifest_bytes)
    if historic['caches']['perp_1m']['content_sha256']!=record['content_sha256']:
        raise ValueError('UM content binding differs from historical manifest')
    return dict(record)


def verify_um_binding(root,document):
    root=Path(root)
    validate_binding(document)
    return verify_um_payload(document,(root/UM_PATH).read_bytes(),
                             (root/HISTORICAL_MANIFEST).read_bytes())


def verify_grid_binding(document,spot_grid,cm_grid,funding):
    """Validate reconstructed grids before any Context construction or cache write."""
    from .data import frame_content_sha256
    validate_binding(document)
    actual={'SPOT':frame_content_sha256(spot_grid),'CM':frame_content_sha256(cm_grid),
            'CM_funding':frame_content_sha256(funding.to_frame())}
    expected={'SPOT':document['SPOT']['grid_content_sha256'],
              'CM':document['CM']['grid_content_sha256'],
              'CM_funding':document['CM_funding']['content_sha256']}
    for key,label in (('SPOT','SPOT grid'),('CM','CM grid'),('CM_funding','CM funding')):
        if actual[key]!=expected[key]:raise ValueError(label+' content differs from input binding')
    return actual

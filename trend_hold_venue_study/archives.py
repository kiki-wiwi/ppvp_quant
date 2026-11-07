"""CP2 raw-byte-only official archive preparation; no ZIP/CSV parsing here."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import urllib.request
from . import scope

BASE='https://data.binance.vision/data/'
FOLDER='reports/trend_hold_venue_round/cp2_registration'

def archive_specs():
    rows=[]
    for venue,kind,start in (('SPOT','klines','2020-01'),('CM','klines','2020-08'),('CM','fundingRate','2022-07')):
        for year in range(2020,2024):
            for month in range(1,13):
                ym=f'{year:04d}-{month:02d}'
                if ym<start:continue
                symbol='BTCUSDT' if venue=='SPOT' else 'BTCUSD_PERP'
                market='spot' if venue=='SPOT' else 'futures/cm'
                relative=(f'{market}/monthly/klines/{symbol}/1m/{symbol}-1m-{ym}.zip' if kind=='klines'
                          else f'{market}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip')
                rows.append(dict(venue=venue,kind=kind,month=ym,relative_path=relative,
                                 path='data/raw/'+relative,url=BASE+relative))
    return rows

def validate_archive(row):
    matches=[r for r in archive_specs() if r['relative_path']==row.get('relative_path')]
    if len(matches)!=1:raise ValueError('archive path outside fixed BTC IS registration')
    expected=matches[0]
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('archive identity/URL mismatch')
    return expected['relative_path']

def checksum_hash(raw,filename):
    text=raw.decode('ascii').strip()
    match=re.fullmatch(r'([0-9a-fA-F]{64})[ \t]+\*?([^\r\n]+)',text)
    if match is None or match.group(2)!=filename:raise ValueError('invalid official SHA256 checksum record')
    return match.group(1).lower()

def _get(url):
    if not url.startswith(BASE):raise ValueError('only official Binance Vision URLs allowed')
    request=urllib.request.Request(url,headers={'User-Agent':'ppvp-quant-preregistered-research/CP2'})
    # No automatic retry. Any network or checksum exception terminates preparation.
    with urllib.request.urlopen(request,timeout=30) as response:return response.read()

def prepare(root):
    if scope.BTC_IS_UNLOCKED or scope.POST_IS_AND_ETH_UNLOCKED:
        raise PermissionError('raw archive preparation requires both stage locks False')
    root=Path(root);out=root/FOLDER
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/'raw_archives.json'
    progress=out/'raw_download_progress.json'
    if manifest.exists() or progress.exists():raise FileExistsError('archive registration attempt exists; no automatic resume')
    stamp=lambda:datetime.now(timezone.utc).isoformat()
    record=dict(status='preparing_raw_bytes',started_at=stamp(),scope='BTC only; <2024-01-01; no CSV parsing/features/signals/accounts',
                archives=[],expected_archives=107,automatic_retry=False)
    def write_progress():progress.write_text(json.dumps(record,indent=2)+'\n')
    write_progress()
    try:
        for spec in archive_specs():
            validate_archive(spec)
            path=root/spec['path'];checksum_path=Path(str(path)+'.CHECKSUM')
            record['current_archive']=spec['relative_path'];write_progress()
            checksum=checksum_path.read_bytes() if checksum_path.exists() else _get(spec['url']+'.CHECKSUM')
            expected=checksum_hash(checksum,path.name)
            data=path.read_bytes() if path.exists() else _get(spec['url'])
            actual=sha256(data).hexdigest()
            if actual!=expected:raise ValueError('official archive SHA256 mismatch: '+spec['relative_path'])
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():path.write_bytes(data)
            if checksum_path.exists() and checksum_path.read_bytes()!=checksum:
                raise ValueError('existing official CHECKSUM bytes differ: '+str(checksum_path))
            if not checksum_path.exists():checksum_path.write_bytes(checksum)
            entry={**spec,'sha256':actual,'bytes':len(data),
                   'checksum_path':str(checksum_path.relative_to(root)),
                   'checksum_sha256':sha256(checksum).hexdigest(),
                   'verified_against':'official CHECKSUM; raw zip bytes only'}
            record['archives'].append(entry);write_progress()
            print(f"verified {len(record['archives'])}/107 {spec['relative_path']} {len(data)} bytes",flush=True)
        record.update(status='raw_archives_registered',ended_at=stamp());record.pop('current_archive',None)
        manifest.write_text(json.dumps(record,indent=2)+'\n')
        progress.unlink()
        return record
    except Exception as exc:
        record.update(status='STOPPED',ended_at=stamp(),error_type=type(exc).__name__,error=str(exc))
        write_progress()
        (out/'STOPPED_raw_preparation.json').write_text(json.dumps(record,indent=2)+'\n')
        raise

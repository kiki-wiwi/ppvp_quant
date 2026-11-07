"""Approved scheme A: official funding-only validation, never market features.

Early authority is bounded official dapi JSON; from July 2022 onward authority
is CHECKSUM-verified Vision CSV, except the explicitly approved 54 month-end
settlements supplied by already saved dapi pages. Alignment matches R5 rounding.
Both raw sources and every overlap difference are retained for audit.
"""
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_EVEN
from hashlib import sha256
import csv
import calendar
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request
from . import scope

REG='reports/trend_hold_venue_round/cp2_registration'
MANIFEST=REG+'/cm_funding.json'
API='https://dapi.binance.com/dapi/v1/fundingRate'
STEP=8*60*60*1000
MINUTE=60000
TOLERANCE=Decimal('1e-8')

def ms(value):return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()*1000)
NATIVE_LOWER=ms('2020-08-11')
COMMON_START=ms('2020-09-01')
OVERLAP_START=ms('2022-07-01')
END=ms('2024-01-01')

def stamp():return datetime.now(timezone.utc).isoformat()
def digest(raw):return sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def canonical_hash(records):return digest(json.dumps(records,separators=(',',':')).encode())

def integer(value):
    d=Decimal(str(value))
    if not d.is_finite() or d!=d.to_integral_value():raise ValueError('finite integer millisecond timestamp required')
    return int(d)

def align_time(value):
    raw=integer(value)
    nearest=int((Decimal(raw)/STEP).to_integral_value(rounding=ROUND_HALF_EVEN))*STEP
    offset=raw-nearest
    if abs(offset)>=MINUTE:raise ValueError('funding timestamp >=60 seconds from UTC 8h boundary')
    rounded=int((Decimal(raw)/MINUTE).to_integral_value(rounding=ROUND_HALF_EVEN))*MINUTE
    if rounded!=nearest:raise ValueError('R5 minute rounding does not land on UTC 8h grid')
    if not NATIVE_LOWER<=rounded<END:raise ValueError('CM funding outside authorized native BTC IS scope')
    return rounded,offset

def rate(value):
    d=Decimal(str(value))
    if not d.is_finite():raise ValueError('finite Decimal funding rate required')
    return str(value)

def unique(rows,label):
    times=[r['time'] for r in rows]
    if not times or times!=sorted(set(times)):raise ValueError(label+': nonempty, ordered and unique settlements required')
    return times

def validate_request(params):
    if set(params)!={'symbol','startTime','endTime','limit'} or params['symbol']!='BTCUSD_PERP' or params['limit']!=1000:
        raise ValueError('only fixed BTCUSD_PERP funding request allowed')
    start,end=integer(params['startTime']),integer(params['endTime'])
    if not NATIVE_LOWER<=start<=end<END:raise ValueError('funding request outside authorized BTC IS')
    return start,end

def parse_api(raw,start,end):
    payload=json.loads(raw)
    if not isinstance(payload,list) or len(payload)>1000:raise ValueError('unexpected funding API payload')
    rows=[]
    for row in payload:
        if not isinstance(row,dict) or row.get('symbol')!='BTCUSD_PERP':raise ValueError('unexpected API symbol')
        original=integer(row['fundingTime'])
        if not start<=original<=end:raise ValueError('API row outside requested raw time boundary')
        time,offset=align_time(original)
        rows.append({'time':time,'rate':rate(row['fundingRate']),'raw_time':original,'offset_ms':offset})
    if rows:unique(rows,'API page')
    return rows

def parse_vision(raw):
    table=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if table.fieldnames!=['calc_time','funding_interval_hours','last_funding_rate']:
        raise ValueError('unexpected official funding CSV columns')
    rows=[]
    for row in table:
        if Decimal(row['funding_interval_hours'])!=8:raise ValueError('non-8h CM funding interval')
        original=integer(row['calc_time']);time,offset=align_time(original)
        rows.append({'time':time,'rate':rate(row['last_funding_rate']),'raw_time':original,'offset_ms':offset})
    unique(rows,'Vision archive')
    return rows

def compare_overlap(api,vision,expected_times=None):
    expected_times=list(range(OVERLAP_START,END,STEP)) if expected_times is None else list(expected_times)
    if unique(api,'API overlap')!=expected_times or unique(vision,'Vision overlap')!=expected_times:
        raise ValueError('overlap missing, extra or misplaced settlement')
    differences=[]
    for a,v in zip(api,vision):
        delta=Decimal(a['rate'])-Decimal(v['rate'])
        differences.append({'funding_time_ms':a['time'],'dapi_rate':a['rate'],'vision_rate':v['rate'],
                            'decimal_diff':str(delta),'abs_decimal_diff':str(abs(delta))})
    maximum=max(abs(Decimal(row['decimal_diff'])) for row in differences)
    exact=sum(Decimal(row['decimal_diff'])==0 for row in differences)
    if maximum>TOLERANCE:raise ValueError(f'CM funding overlap absolute Decimal difference {maximum} > 1e-8')
    return {'settlements':len(differences),'exact_decimal_match_count':exact,
            'exact_decimal_match_ratio':str(Decimal(exact)/len(differences)),
            'within_authorized_tail_tolerance_count':len(differences)-exact,
            'max_abs_decimal_diff':str(maximum),'tolerance':'1e-8','comparisons':differences}

def approved_month_end_times():
    times=[]
    for year in (2022,2023):
        for month in range(7 if year==2022 else 1,13):
            day=calendar.monthrange(year,month)[1]
            start=ms(f'{year:04d}-{month:02d}-{day:02d}')
            times.extend(start+offset*STEP for offset in range(3))
    return times

def assemble(api,vision):
    api_times=unique(api,'all API funding')
    vision_times=unique(vision,'all Vision funding')
    expected=list(range(COMMON_START,END,STEP))
    if [t for t in api_times if t>=COMMON_START]!=expected:raise ValueError('IS-Common requires exactly 3651 continuous 8h settlements')
    august=[t for t in api_times if t<COMMON_START]
    if not august or august!=list(range(august[0],COMMON_START,STEP)):
        raise ValueError('August native funding missing or discontinuous from first observed settlement')
    overlap_times=list(range(OVERLAP_START,END,STEP))
    approved=set(approved_month_end_times())
    existing=[t for t in overlap_times if t not in approved]
    if len(approved)!=54 or len(existing)!=1593 or vision_times!=existing:
        raise ValueError('Vision gap differs from the exact 54 approved month-end settlements')
    overlap=compare_overlap([r for r in api if r['time'] in set(existing)],vision,expected_times=existing)
    api_map={r['time']:r for r in api};vision_map={r['time']:r for r in vision}
    canonical=[]
    for time in api_times:
        if time<OVERLAP_START:source,row='dapi',api_map[time]
        elif time in approved:source,row='dapi_month_end_supplement',api_map[time]
        else:source,row='vision',vision_map[time]
        canonical.append({'funding_time_ms':time,'funding_rate':row['rate'],'source':source,
                          'raw_funding_time_ms':row['raw_time'],'source_path':row.get('source_path')})
    if [r['funding_time_ms'] for r in canonical if r['funding_time_ms']>=OVERLAP_START]!=overlap_times:
        raise ValueError('supplemented overlap is not exactly 1647 unique 8h settlements')
    return {'common_settlements':len(expected),'overlap':overlap,
            'supplemented_overlap_settlements':len(overlap_times),
            'month_end_supplement_count':54,
            'month_end_supplements':[r for r in canonical if r['source']=='dapi_month_end_supplement'],
            'august_launch':{'first_settlement_ms':august[0],'last_settlement_ms':august[-1],
                            'settlements':len(august),'continuous_from_first_settlement':True},
            'maximum_abs_timestamp_offset_ms':max(abs(r['offset_ms']) for r in api+vision),
            'canonical_records':canonical,'canonical_sha256':canonical_hash(canonical)}

def _vision(root,manifest):
    from .data import validate_archive_manifest,_read_archive
    result=[]
    for entry in validate_archive_manifest(manifest):
        if entry['kind']!='fundingRate':continue
        rows=parse_vision(_read_archive(root,entry))
        for row in rows:row['source_path']=entry['path']
        year,month=map(int,entry['month'].split('-'))
        start=ms(f'{year:04d}-{month:02d}-01')
        following=ms(f'{year+1:04d}-01-01') if month==12 else ms(f'{year:04d}-{month+1:02d}-01')
        if any(not start<=r['time']<following for r in rows):raise ValueError('Vision funding outside registered month')
        result.extend(rows)
    return result

def prepare(root):
    if scope.BTC_IS_UNLOCKED or scope.POST_IS_AND_ETH_UNLOCKED:raise PermissionError('funding preparation requires both locks False')
    root=Path(root);folder=root/REG;target=root/MANIFEST;progress=folder/'funding_progress.json'
    if target.exists() or progress.exists():raise FileExistsError('funding preparation attempt exists; no automatic retry')
    archive_raw=(folder/'raw_archives.json').read_bytes()
    archive_manifest=json.loads(archive_raw)
    pages=folder/'funding_api_pages';pages.mkdir(exist_ok=False)
    record={'status':'preparing','started_at':stamp(),'api':API,'symbol':'BTCUSD_PERP',
            'raw_archive_manifest_sha256':digest(archive_raw),'api_pages':[],
            'authorization':'Official scheme A; funding tables only before BTC unlock; no OHLC/features/accounts',
            'automatic_retry':False}
    progress.write_bytes(encoded(record))
    try:
        all_api=[]
        for year in range(2020,2024):
            for month in range(1,13):
                if year==2020 and month<8:continue
                start=max(NATIVE_LOWER,ms(f'{year:04d}-{month:02d}-01'))
                end=(ms(f'{year+1:04d}-01-01') if month==12 else ms(f'{year:04d}-{month+1:02d}-01'))-1
                page_number=0
                while start<=end:
                    params={'symbol':'BTCUSD_PERP','startTime':start,'endTime':end,'limit':1000}
                    validate_request(params)
                    url=API+'?'+urllib.parse.urlencode(params)
                    requested_at=stamp()
                    request=urllib.request.Request(url,headers={'User-Agent':'ppvp-quant-preregistered-research/CP2'})
                    with urllib.request.urlopen(request,timeout=30) as response:raw=response.read()
                    page_number+=1
                    path=pages/f'{year:04d}-{month:02d}_page_{page_number:03d}.json'
                    with path.open('xb') as output:output.write(raw)
                    entry={'path':str(path.relative_to(root)),'sha256':digest(raw),'bytes':len(raw),
                           'url':url,'parameters':params,'requested_at':requested_at,'received_at':stamp()}
                    record['api_pages'].append(entry);progress.write_bytes(encoded(record))
                    parsed=parse_api(raw,start,end)
                    for row in parsed:row['source_path']=entry['path']
                    entry['rows']=len(parsed);all_api.extend(parsed);progress.write_bytes(encoded(record))
                    print(f"funding API {year:04d}-{month:02d} page {page_number}: {len(parsed)} records",flush=True)
                    if len(parsed)<1000:break
                    start=parsed[-1]['raw_time']+1
        vision=_vision(root,archive_manifest)
        result=assemble(all_api,vision)
        record.update(result,status='validated',ended_at=stamp(),authority_split_ms=OVERLAP_START)
        target.write_bytes(encoded(record));progress.unlink()
        return record
    except Exception as exc:
        record.update(status='STOPPED',ended_at=stamp(),error_type=type(exc).__name__,error=str(exc))
        progress.write_bytes(encoded(record));(folder/'STOPPED_funding.json').write_bytes(encoded(record))
        raise

def verify_registered(root):
    """Rebuild registered funding tables from bound bytes, with no network calls."""
    if scope.POST_IS_AND_ETH_UNLOCKED:raise PermissionError('CP2 requires POST IS/ETH locked')
    root=Path(root);document=json.loads((root/MANIFEST).read_text())
    if document.get('status')!='validated':raise ValueError('funding registration not validated')
    raw=(root/REG/'raw_archives.json').read_bytes()
    if digest(raw)!=document['raw_archive_manifest_sha256']:raise ValueError('funding archive registration changed')
    api=_saved_api(root,document)
    rebuilt=assemble(api,_vision(root,json.loads(raw)))
    for field in rebuilt:
        if rebuilt[field]!=document[field]:raise ValueError('funding registration reconstruction differs: '+field)
    import pandas as pd
    records=rebuilt['canonical_records']
    index=pd.DatetimeIndex(pd.to_datetime([r['funding_time_ms'] for r in records],unit='ms',utc=True)).as_unit('ns')
    index.name='funding_time'
    return pd.Series([float(Decimal(r['funding_rate'])) for r in records],index=index,name='funding_rate')


def _saved_api(root,document):
    api=[]
    for entry in document['api_pages']:
        start,end=validate_request(entry['parameters'])
        if entry['url']!=API+'?'+urllib.parse.urlencode(entry['parameters']):raise ValueError('registered API URL mismatch')
        rel=Path(entry['path'])
        if rel.parent!=Path(REG)/'funding_api_pages' or rel.suffix!='.json':raise ValueError('unregistered API page path')
        payload=(root/rel).read_bytes()
        if digest(payload)!=entry['sha256']:raise ValueError('API raw page hash mismatch')
        parsed=parse_api(payload,start,end)
        for row in parsed:row['source_path']=entry['path']
        api.extend(parsed)
    return api


def register_saved(root):
    """Explicit approved resume from preserved bytes; never calls the network."""
    if scope.BTC_IS_UNLOCKED or scope.POST_IS_AND_ETH_UNLOCKED:
        raise PermissionError('saved funding registration requires both locks False')
    root=Path(root);folder=root/REG;target=root/MANIFEST
    if target.exists():raise FileExistsError('funding registration exists; no automatic overwrite')
    prior=folder/'validation_history/vision_month_end_stop/STOPPED_funding.json'
    original=prior.read_bytes();record=json.loads(original)
    record['saved_attempt_path']=str(prior.relative_to(root))
    record['saved_attempt_sha256']=digest(original)
    record['resume_started_at']=stamp()
    record['resume_authorization']='Only the 54 known 2022-07..2023-12 month-end settlements use saved dapi; compare all 1593 Vision rates first with <=1e-8.'
    record['resume_network_requests']=0
    try:
        archive_raw=(folder/'raw_archives.json').read_bytes()
        if digest(archive_raw)!=record['raw_archive_manifest_sha256']:raise ValueError('raw archive manifest changed')
        api=_saved_api(root,record)
        result=assemble(api,_vision(root,json.loads(archive_raw)))
        for key in ('error_type','error'):record.pop(key,None)
        record.update(result,status='validated',ended_at=stamp(),authority_split_ms=OVERLAP_START)
        with target.open('xb') as output:output.write(encoded(record))
        return record
    except Exception as exc:
        record.update(status='STOPPED',ended_at=stamp(),error_type=type(exc).__name__,error=str(exc))
        with (folder/'STOPPED_saved_funding.json').open('xb') as output:output.write(encoded(record))
        raise

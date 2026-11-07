"""Authorized input-only preflight. No market writes, network, features or accounts."""
from pathlib import Path
import sys,os,json,hashlib,traceback,gc
from datetime import datetime,timezone
import importlib.abc
ROOT=Path('/home/ubuntu/.codex/worktrees/channel-vah-prereg/ppvp_quant')
sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/trend_hold_venue_round/cp2_registration/input_revision_2/preflight'
BLOCKED=('trend_hold_venue_study.features','trend_hold_venue_study.portfolios','trend_hold_venue_study.linear',
 'trend_hold_venue_study.runtime','trend_hold_venue_study.statistics','trend_hold_study.features',
 'trend_hold_study.engine','trend_hold_study.statistics','impulse_study.engine','impulse_study.features')
class NoStrategyImports(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in BLOCKED:raise PermissionError('Strategy module prohibited during input preflight: '+fullname)
sys.meta_path.insert(0,NoStrategyImports())
def audit(event,args):
 if event=='socket.connect':raise PermissionError('Network prohibited in readonly preflight')
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  path=os.fsdecode(args[0]).replace('\\','/')
  flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
  if '/data/' in '/'+path and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
   raise PermissionError('Market/cache write prohibited: '+path)
sys.addaudithook(audit)
import numpy as np
import pandas as pd
from impulse_study.data_1m import content_sha256
from trend_hold_venue_study import data,scope,cp2_control as control,input_binding

def stamp():return datetime.now(timezone.utc).isoformat()
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def save(name,value):
 with (OUT/name).open('w') as stream:json.dump(value,stream,indent=2,allow_nan=False)
def require_locked():
 if scope.BTC_IS_UNLOCKED or scope.POST_IS_AND_ETH_UNLOCKED:raise PermissionError('Both locks must remain False')
require_locked()
OUT.mkdir(exist_ok=False)
summary={'started_at':stamp(),'status':'running','authorization':'BTC IS input content/structure only while both locks False; no features, signals, budgets or accounts',
 'script_path':str(Path(__file__)),'script_sha256':digest(__file__),'network_requests':0,'market_files_written':0,
 'strategy_modules_blocked':list(BLOCKED),'archives':[],'venues':{}}
raw_manifest=ROOT/data.MANIFEST
original_manifest=ROOT/'data/manifest_impulse_1m.json'
original_manifest_hash=digest(original_manifest)
source_state=control.verify_frozen(ROOT,btc_unlocked=False)
summary['frozen_before']=source_state
um_path=ROOT/'data/processed_1m/BTCUSDT_perp_1m_IS.parquet'
um_byte_hash=digest(um_path)
try:
 manifest=json.loads(original_manifest.read_text());expected=manifest['caches']['perp_1m']
 if expected['path']!='data/processed_1m/BTCUSDT_perp_1m_IS.parquet':raise ValueError('Unexpected UM cache path')
 frame=pd.read_parquet(um_path,filters=[('open_time','>=',data.IS_START),('open_time','<',data.IS_END)])
 frame.index=pd.DatetimeIndex(frame.index).tz_convert('UTC').as_unit('ns')
 if frame.index[0]!=data.IS_START or frame.index[-1]!=data.IS_END-pd.Timedelta(minutes=1) or len(frame)!=2103840:
  raise ValueError('UM IS cache range/count mismatch')
 actual=content_sha256(frame)
 um={'status':'passed' if actual==expected['content_sha256'] else 'STOPPED','path':expected['path'],
  'rows':len(frame),'first_time':str(frame.index[0]),'last_time':str(frame.index[-1]),
  'historical_manifest_path':str(original_manifest.relative_to(ROOT)),
  'historical_manifest_sha256':original_manifest_hash,'historical_parquet_sha256':expected['parquet_sha256'],
  'current_parquet_sha256':um_byte_hash,'registered_content_sha256':expected['content_sha256'],'actual_content_sha256':actual,
  'container_byte_difference_with_identical_registered_content':um_byte_hash!=expected['parquet_sha256'] and actual==expected['content_sha256'],
  'formal_input_binding_required':{'parquet_sha256':um_byte_hash,'content_sha256':expected['content_sha256']},
  'interpretation':'Same frozen column/index contents; Parquet serialization/container bytes differ. No cache or historical manifest is rewritten.'}
 save('um_content_registration.json',um);summary['UM']=um
 if actual!=expected['content_sha256']:raise ValueError('UM frozen content SHA256 mismatch')
 print('UM content hash exact; current Parquet bytes and historical manifest hash registered.',flush=True)
 del frame;gc.collect()
 entries=data.validate_archive_manifest(json.loads(raw_manifest.read_text()))
 for venue,total in [('SPOT',48),('CM',41)]:
  parts=[];exception_records=[];selected=[e for e in entries if e['venue']==venue and e['kind']=='klines']
  if len(selected)!=total:raise ValueError('Wrong authorized archive count')
  for entry in selected:
   require_locked();raw=data._read_archive(ROOT,entry)
   first=raw.splitlines()[0].split(b',')[0].strip().lstrip(b'\xef\xbb\xbf')
   has_header=not first.isdigit()
   native=data.parse_klines(raw,entry['month'],venue)
   exceptions=native.attrs.get('raw_close_time_exceptions',[]);exception_records.extend(exceptions)
   first_data=raw.splitlines()[1 if has_header else 0].split(b',')[0]
   units='us' if int(first_data)>10**14 else 'ms'
   item={'venue':venue,'month':entry['month'],'path':entry['path'],'sha256':entry['sha256'],
    'official_checksum_sha256':entry['checksum_sha256'],'checksum_verified':True,
    'has_header':has_header,'timestamp_unit':units,'normalized_timestamp_unit':'UTC ns','columns':12,
    'rows':len(native),'first_open_time':native.index[0].isoformat(),'last_open_time':native.index[-1].isoformat(),
    'native_OHLCV_content_sha256':data.frame_content_sha256(native),
    'raw_close_time_exceptions':exceptions,'OHLCV_finite_positive_prices_validated':True}
   summary['archives'].append(item);parts.append(native)
  native=pd.concat(parts);data._ordered(native.index,venue+' native')
  if venue=='CM':
   daily,daily_records=input_binding.load_daily_supplements(ROOT)
   native=input_binding.merge_cm_supplements(native,daily)
   summary['CM_daily_supplements']=daily_records
   summary['CM_daily_supplement_count']=len(daily)
   del daily
  start=data.IS_START if venue=='SPOT' else data.cm_native_start(native)
  native=native.loc[start:]
  expected_index=pd.date_range(start,data.IS_END,inclusive='left',freq='min').as_unit('ns')
  placeholder=(native['volume']==0)&(native['open']==native['high'])&(native['high']==native['low'])&(native['low']==native['close'])
  valid=pd.Series(~placeholder.to_numpy(),index=native.index).reindex(expected_index,fill_value=False)
  effective_daily=valid.groupby(valid.index.floor('D')).sum()
  full_days=effective_daily.loc[(effective_daily.index>=start.ceil('D'))&(effective_daily.index<data.IS_END)]
  empty=full_days[full_days==0]
  coverage={'venue':venue,'archives_verified':len(selected),'native_rows':len(native),'grid_minutes':len(expected_index),
   'first_grid_minute':start.isoformat(),'last_grid_minute':expected_index[-1].isoformat(),
   'native_OHLCV_content_sha256':data.frame_content_sha256(native),
   'archive_gap_count':len(expected_index)-len(native),'placeholder_count':int(placeholder.sum()),
   'bad_count':int((~valid).sum()),'full_UTC_days_checked':len(full_days),'zero_effective_days':[t.isoformat() for t in empty.index],
   'raw_close_time_exception_count':len(exception_records)}
  if venue=='SPOT':coverage['approved_timestamp_exceptions']=data.validate_clock_exception_inventory(exception_records)
  summary['venues'][venue]=coverage
  save('native_archive_registration.json',{'sha256_algorithm':'frame_content_sha256 (ordered UTC-ns index and little-endian OHLCV arrays)',
       'archives':summary['archives'],'venues':summary['venues']})
  if len(empty):
   coverage['status']='STOPPED';coverage['grid_content_sha256']=None
   coverage['reason']='No admissible completed grid: complete UTC day has no effective minute; no cache constructed or saved'
   save(venue.lower()+'_structure_verification.json',coverage)
   raise ValueError(venue+': complete UTC day has no effective minute: '+', '.join(t.strftime('%Y-%m-%d') for t in empty.index))
  grid=data.build_grid(native,start,data.IS_END)
  if not grid.index.equals(expected_index):raise ValueError(venue+': completed grid index mismatch')
  if not np.isfinite(grid.loc[:,list(data.OHLCV)].to_numpy()).all() or (grid.loc[:,list(data.OHLCV[:4])].to_numpy()<=0).any():
   raise ValueError(venue+': completed grid has invalid prices')
  coverage.update(status='passed',grid_content_sha256=data.frame_content_sha256(grid),
   deterministic_grid_rule='existing bad = missing OR flat zero-volume placeholder; archive_gap=missing; prior-only positive price fill')
  save(venue.lower()+'_structure_verification.json',coverage)
  print(f'{venue}: {total} archives parsed; grid {len(grid)} minutes; bad={coverage["bad_count"]}; archive_gap={coverage["archive_gap_count"]}; no zero-effective day.',flush=True)
  if venue=='SPOT':spot_grid=grid
  else:cm_grid=grid
  del native,parts,valid,placeholder;gc.collect()
 funding=data.funding_for_context(__import__('trend_hold_venue_study.funding_sources',fromlist=['verify_registered']).verify_registered(ROOT),cm_grid.index[0])
 binding={'UM':{'path':expected['path'],'current_parquet_sha256':um_byte_hash,'content_sha256':actual,'historical_manifest_sha256':original_manifest_hash},
  'SPOT':{'grid_content_sha256':summary['venues']['SPOT']['grid_content_sha256']},
  'CM':{'grid_content_sha256':summary['venues']['CM']['grid_content_sha256']},
  'CM_funding':{'content_sha256':data.frame_content_sha256(funding.to_frame())}}
 input_binding.verify_um_binding(ROOT,binding)
 input_binding.verify_grid_binding(binding,spot_grid,cm_grid,funding)
 binding_path=OUT.parent/'input_binding.json'
 with binding_path.open('x') as output:json.dump(binding,output,indent=2)
 summary['input_binding']=binding
 summary['CM_funding']={'canonical_records':funding.attrs['canonical_record_count'],'context_records':len(funding),'context_exclusions':funding.attrs['context_funding_exclusions']}
 summary['status']='passed'
except Exception as exc:
 summary.update(status='STOPPED',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
 print(type(exc).__name__+': '+str(exc),flush=True)
finally:
 require_locked()
 summary['ended_at']=stamp()
 summary['cache_bytes_unchanged']=digest(um_path)==um_byte_hash
 summary['historical_manifest_unchanged']=digest(original_manifest)==original_manifest_hash
 summary['frozen_after']=control.verify_frozen(ROOT,btc_unlocked=False)
 summary['CP2_registration_stage']='Revised file hashes are registered only after input preflight and synthetic validation pass; historical manifests unchanged'
 summary['strategy_modules_loaded']=[m for m in BLOCKED if m in sys.modules]
 if not summary['cache_bytes_unchanged'] or not summary['historical_manifest_unchanged'] or summary['strategy_modules_loaded']:
  summary.update(status='STOPPED',integrity_error=True)
 save('preflight_summary.json',summary)
 if summary['status']=='STOPPED':save('STOPPED_preflight.json',summary)
 hashes={str(p.relative_to(ROOT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='evidence_hashes.json'}
 save('evidence_hashes.json',{'sha256':hashes,'external_preflight_script_sha256':digest(__file__)})
 print(json.dumps({'status':summary['status'],'archives_verified':len(summary['archives']),
   'UM_content_match':summary.get('UM',{}).get('status')=='passed','venues':summary['venues'],
   'both_locks_false':not scope.BTC_IS_UNLOCKED and not scope.POST_IS_AND_ETH_UNLOCKED,
   'cache_bytes_unchanged':summary['cache_bytes_unchanged'],'historical_manifest_unchanged':summary['historical_manifest_unchanged'],
   'strategy_modules_loaded':summary['strategy_modules_loaded']},indent=2),flush=True)
if summary['status']!='passed':sys.exit(1)

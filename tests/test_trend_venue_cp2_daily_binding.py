"""Approved four-day supplement and input binding; supplied synthetic bytes only."""
import copy
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from trend_hold_venue_study import data
from trend_hold_venue_study import input_binding as binding


DAYS=('2023-08-28','2023-08-29','2023-08-30','2023-08-31')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture(day, *, omit=False, duplicate=False, shift=False):
    index=pd.date_range(day,periods=1440,freq='min',tz='UTC').as_unit('ns')
    if omit:index=index[:-1]
    if duplicate:index=index[:-1].append(index[-2:-1])
    if shift:index=index+pd.Timedelta(minutes=1)
    lines=[]
    for stamp in index.asi8//1_000_000:
        lines.append(f'{stamp},100,101,99,100.5,2,{stamp+59999},201,2,1,100,0')
    raw=('\n'.join(lines)+'\n').encode()
    name=f'BTCUSD_PERP-1m-{day}.zip'
    relative=f'futures/cm/daily/klines/BTCUSD_PERP/1m/{name}'
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as archive:
        archive.writestr(name.removesuffix('.zip')+'.csv',raw)
    blob=stream.getvalue()
    checksum=f'{digest(blob)}  {name}\n'.encode()
    # The normal fixture content is independent of the production CSV parser.
    expected=pd.DataFrame(np.tile([100.,101.,99.,100.5,2.],(len(index),1)),index=index,
                          columns=['open','high','low','close','volume'])
    expected.index.name='open_time'
    entry={'venue':'CM','kind':'klines','day':day,'path':'data/raw/'+relative,
           'url':'https://data.binance.vision/data/'+relative,'sha256':digest(blob),
           'checksum_path':'data/raw/'+relative+'.CHECKSUM','checksum_sha256':digest(checksum),
           'rows':1440,'content_sha256':data.frame_content_sha256(expected)}
    return entry,blob,checksum,expected


def four_days():
    items=[fixture(day) for day in DAYS]
    return items,pd.concat([x[3] for x in items])


def test_four_exact_daily_packages_preserve_all_native_values_and_minutes():
    items,expected=four_days()
    entries=binding.validate_daily_manifest({'archives':[x[0] for x in reversed(items)]})
    assert [entry['day'] for entry in entries]==list(DAYS)
    actual=pd.concat([binding.parse_daily_payload(entry,blob,checksum) for entry,blob,checksum,_ in items])
    pd.testing.assert_frame_equal(actual,expected,check_exact=True,check_freq=False)
    assert len(actual)==5760 and not actual.index.has_duplicates
    monthly=expected.iloc[:0].copy()
    start=pd.Timestamp('2023-08-27T23:59Z')
    monthly.loc[start]=[98.,100.,97.,99.,3.]
    before=monthly.copy(deep=True)
    merged=binding.merge_cm_supplements(monthly,actual)
    assert len(merged)==5761 and merged.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(merged.loc[expected.index],expected,check_exact=True,check_freq=False)
    pd.testing.assert_frame_equal(monthly,before,check_exact=True)


@pytest.mark.parametrize('mode',['omit','duplicate','shift'])
def test_daily_package_rejects_missing_duplicate_or_off_day_minutes(mode):
    entry,blob,checksum,_=fixture(DAYS[0],**{mode:True})
    with pytest.raises(ValueError):binding.parse_daily_payload(entry,blob,checksum)


def test_daily_manifest_rejects_wrong_day_duplicate_and_unofficial_path():
    items,_=four_days();entries=[x[0] for x in items]
    for candidate in (entries[:-1],entries+[entries[0]]):
        with pytest.raises(ValueError,match='exact four'):
            binding.validate_daily_manifest({'archives':candidate})
    for field,value in [('day','2023-09-01'),('venue','SPOT'),('kind','fundingRate'),
                        ('path','../outside.zip'),('url','https://example.org/file.zip'),('rows',1439)]:
        changed=copy.deepcopy(entries);changed[0][field]=value
        with pytest.raises(ValueError):binding.validate_daily_manifest({'archives':changed})


def test_daily_payload_rejects_archive_checksum_content_and_identity_tampering():
    entry,blob,checksum,_=fixture(DAYS[0])
    with pytest.raises(ValueError,match='archive hash'):
        binding.parse_daily_payload(entry,blob+b'changed',checksum)
    with pytest.raises(ValueError,match='checksum hash'):
        binding.parse_daily_payload(entry,blob,checksum+b'changed')
    bad=copy.deepcopy(entry);bad['content_sha256']='0'*64
    with pytest.raises(ValueError,match='content hash'):
        binding.parse_daily_payload(bad,blob,checksum)
    bad_checksum=(entry['sha256']+'  WRONG.zip\n').encode()
    bad=copy.deepcopy(entry);bad['checksum_sha256']=digest(bad_checksum)
    with pytest.raises(ValueError,match='bind'):
        binding.parse_daily_payload(bad,blob,bad_checksum)


def test_merge_rejects_any_overlap_or_incomplete_supplement_without_replacing_monthly():
    _,daily=four_days();monthly=daily.iloc[:1].copy();before=monthly.copy(deep=True)
    with pytest.raises(ValueError,match='overlap'):
        binding.merge_cm_supplements(monthly,daily)
    pd.testing.assert_frame_equal(monthly,before,check_exact=True)
    for candidate in (daily.iloc[:-1],pd.concat([daily,daily.iloc[-1:]])):
        with pytest.raises(ValueError):binding.merge_cm_supplements(monthly.iloc[:0],candidate)
    extra=daily.iloc[-1:].copy();extra.index=extra.index+pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        binding.merge_cm_supplements(monthly.iloc[:0],pd.concat([daily,extra]))


def binding_fixture():
    parquet=b'synthetic parquet bytes; never decoded or used as market input'
    manifest=json.dumps({'caches':{'perp_1m':{'content_sha256':'c'*64}}},sort_keys=True).encode()
    document={'UM':{'path':'data/processed_1m/BTCUSDT_perp_1m_IS.parquet',
                    'current_parquet_sha256':digest(parquet),'content_sha256':'c'*64,
                    'historical_manifest_sha256':digest(manifest)},
              'SPOT':{'grid_content_sha256':'a'*64},'CM':{'grid_content_sha256':'b'*64},
              'CM_funding':{'content_sha256':'d'*64}}
    return document,parquet,manifest


def test_um_binding_verifies_current_bytes_and_old_manifest_without_mutation(tmp_path):
    document,parquet,manifest=binding_fixture()
    # Files deliberately avoid every guarded market directory; payload checking is pure.
    cache_file=tmp_path/'synthetic_cache.bin';cache_file.write_bytes(parquet)
    manifest_file=tmp_path/'synthetic_history.json';manifest_file.write_bytes(manifest)
    before=copy.deepcopy(document)
    result=binding.verify_um_payload(document,cache_file.read_bytes(),manifest_file.read_bytes())
    assert result['current_parquet_sha256']==digest(parquet)
    assert result['content_sha256']=='c'*64
    assert document==before and manifest_file.read_bytes()==manifest
    with pytest.raises(ValueError,match='UM parquet'):
        binding.verify_um_payload(document,parquet+b'change',manifest)
    with pytest.raises(ValueError,match='historical manifest'):
        binding.verify_um_payload(document,parquet,manifest+b' ')
    bad=copy.deepcopy(document);bad['UM']['content_sha256']='e'*64
    with pytest.raises(ValueError,match='UM content'):
        binding.verify_um_payload(bad,parquet,manifest)
    bad=copy.deepcopy(document);bad['UM']['path']='data/processed_1m/ETHUSDT.parquet'
    with pytest.raises(ValueError,match='UM path'):
        binding.verify_um_payload(bad,parquet,manifest)


def test_grid_bindings_detect_value_index_and_funding_changes_before_context_or_cache():
    document,_,_=binding_fixture()
    _,_,_,native=fixture(DAYS[0])
    grid=data.build_grid(native,native.index[0],native.index[-1]+pd.Timedelta(minutes=1))
    funding=pd.Series([.0001,-.0002],index=native.index[[0,480]],name='funding_rate')
    document['SPOT']['grid_content_sha256']=data.frame_content_sha256(grid)
    document['CM']['grid_content_sha256']=data.frame_content_sha256(grid)
    document['CM_funding']['content_sha256']=data.frame_content_sha256(funding.to_frame())
    checked=binding.verify_grid_binding(document,grid,grid.copy(deep=True),funding)
    assert checked['SPOT']==document['SPOT']['grid_content_sha256']
    changed=grid.copy(deep=True);changed.iloc[1,3]+=.01
    with pytest.raises(ValueError,match='SPOT'):
        binding.verify_grid_binding(document,changed,grid,funding)
    moved=grid.copy(deep=True);moved.index=moved.index+pd.Timedelta(minutes=1)
    with pytest.raises(ValueError,match='CM grid'):
        binding.verify_grid_binding(document,grid,moved,funding)
    rates=funding.copy(deep=True);rates.iloc[0]+=.00001
    with pytest.raises(ValueError,match='CM funding'):
        binding.verify_grid_binding(document,grid,grid,rates)

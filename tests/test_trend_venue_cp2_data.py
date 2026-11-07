"""Synthetic bytes and grids only; no official archive is opened by this suite."""
import io
import zipfile
import numpy as np
import pandas as pd
import pytest
from trend_hold_venue_study import data, scope


def kline_bytes(times, *, header=False, micro=False, rows=None):
    lines=[]
    if header:
        lines.append('open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote,ignore')
    for i,t in enumerate(times):
        stamp=pd.Timestamp(t).value//(1000 if micro else 1_000_000)
        close=stamp+(60_000_000-1 if micro else 60_000-1)
        values=(rows or {}).get(i,(100.,101.,99.,100.,1.))
        lines.append(','.join(map(str,(stamp,*values,close,100.,1,0.5,50.,0))))
    return ('\n'.join(lines)+'\n').encode()


def funding_bytes(times, rates=None, *, micro=False, header=True):
    lines=['calc_time,funding_interval_hours,last_funding_rate'] if header else []
    for i,t in enumerate(times):
        stamp=pd.Timestamp(t).value//(1000 if micro else 1_000_000)
        lines.append(f'{stamp},8,{(rates or [.0001]*len(times))[i]}')
    return ('\n'.join(lines)+'\n').encode()


@pytest.mark.parametrize('header,micro',[(False,False),(True,False),(False,True),(True,True)])
def test_klines_native_ms_us_and_header_share_exact_utc_ns(header,micro):
    times=['2020-01-01T00:00Z','2020-01-01T00:01Z']
    result=data.parse_klines(kline_bytes(times,header=header,micro=micro),'2020-01','SPOT')
    assert result.index.equals(pd.DatetimeIndex(times).as_unit('ns'))
    assert result.index.unit=='ns'
    assert result.open.tolist()==[100.,100.]
    assert result.volume.dtype==np.dtype('float64')


def test_klines_reject_duplicates_outside_month_nonfinite_and_malformed_ohlc():
    t='2020-01-01T00:00Z'
    with pytest.raises(ValueError,match='unique'):
        data.parse_klines(kline_bytes([t,t]),'2020-01','SPOT')
    with pytest.raises(ValueError,match='month'):
        data.parse_klines(kline_bytes(['2020-02-01T00:00Z']),'2020-01','SPOT')
    with pytest.raises(ValueError,match='finite'):
        data.parse_klines(kline_bytes([t],rows={0:(100.,101.,99.,100.,np.nan)}),'2020-01','SPOT')
    with pytest.raises(ValueError,match='OHLC'):
        data.parse_klines(kline_bytes([t],rows={0:(100.,99.,98.,100.,1.)}),'2020-01','SPOT')
    with pytest.raises(ValueError,match='minute'):
        data.parse_klines(kline_bytes(['2020-01-01T00:00:01Z']),'2020-01','SPOT')
    for month,venue in [('2024-01','SPOT'),('2020-01','ETH'),('2020-07','CM')]:
        with pytest.raises(ValueError):data.parse_klines(kline_bytes([t]),month,venue)


@pytest.mark.parametrize('micro,header',[(False,True),(True,False)])
def test_funding_rounds_official_jitter_and_retains_actual_calendar(micro,header):
    raw=funding_bytes(['2020-08-11T16:00:00.002Z','2020-08-12T00:00:00.001Z'],[-.001,.002],micro=micro,header=header)
    result=data.parse_funding(raw,'2020-08')
    expected=pd.DatetimeIndex(['2020-08-11T16:00Z','2020-08-12T00:00Z']).as_unit('ns')
    assert result.index.equals(expected)
    assert result.tolist()==[-.001,.002]


def test_funding_duplicate_after_round_scope_and_nonfinite_rejected():
    with pytest.raises(ValueError,match='unique'):
        data.parse_funding(funding_bytes(['2020-08-11T16:00:00.001Z','2020-08-11T16:00:00.002Z']),'2020-08')
    with pytest.raises(ValueError,match='month'):
        data.parse_funding(funding_bytes(['2024-01-01T00:00Z']),'2023-12')
    with pytest.raises(ValueError,match='finite'):
        data.parse_funding(funding_bytes(['2020-08-11T16:00Z'],[np.inf]),'2020-08')


def test_grid_marks_missing_separately_from_maintenance_and_only_forward_fills():
    times=['2020-01-01T00:00Z','2020-01-01T00:02Z','2020-01-01T00:03Z']
    raw=data.parse_klines(kline_bytes(times,rows={1:(120.,120.,120.,120.,0.),2:(105.,110.,100.,107.,0.)}),'2020-01','SPOT')
    start=pd.Timestamp('2020-01-01T00:00Z');end=start+pd.Timedelta(minutes=4)
    grid=data.build_grid(raw,start,end)
    assert grid.bad.tolist()==[False,True,True,False]
    assert grid.archive_gap.tolist()==[False,True,False,False]
    assert grid.close.tolist()==[100.,100.,100.,107.]
    assert grid.volume.tolist()==[1.,0.,0.,0.]
    assert grid.loc[grid.bad,['open','high','low','close']].to_numpy().tolist()==[[100.]*4,[100.]*4]
    with pytest.raises(ValueError,match='initial'):
        data.build_grid(raw.iloc[1:],start,end)


def test_complete_day_without_effective_minute_is_stop():
    start=pd.Timestamp('2020-01-01T00:00Z')
    raw=data.parse_klines(kline_bytes([start.isoformat()]),'2020-01','SPOT')
    with pytest.raises(ValueError,match='complete UTC day'):
        data.build_grid(raw,start,start+pd.Timedelta(days=2))


def test_cm_native_start_never_fabricates_prelisting():
    raw=data.parse_klines(kline_bytes(['2020-08-11T15:00Z','2020-08-11T15:01Z']),'2020-08','CM')
    assert data.cm_native_start(raw)==pd.Timestamp('2020-08-11T15:00Z')
    early=data.parse_klines(kline_bytes(['2020-08-10T23:59Z']),'2020-08','CM')
    late=data.parse_klines(kline_bytes(['2020-08-12T00:01Z']),'2020-08','CM')
    for frame in (early,late):
        with pytest.raises(ValueError,match='native'):data.cm_native_start(frame)


def test_zip_reader_requires_one_expected_csv_without_extracting():
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:z.writestr('BTCUSDT-1m-2020-01.csv',b'synthetic')
    assert data.unzip_csv(stream.getvalue(),'BTCUSDT-1m-2020-01.zip')==b'synthetic'
    bad=io.BytesIO()
    with zipfile.ZipFile(bad,'w') as z:
        z.writestr('a.csv',b'a');z.writestr('b.csv',b'b')
    with pytest.raises(ValueError,match='single'):
        data.unzip_csv(bad.getvalue(),'BTCUSDT-1m-2020-01.zip')


def test_raw_manifest_requires_exact_complete_allowed_archive_set():
    expected=data.expected_archives()
    assert len(expected)==107
    entries=[]
    for venue,kind,month,rel in expected:
        entries.append(dict(venue=venue,kind=kind,month=month,relative_path=rel,
            path='data/raw/'+rel,url=data.VISION_BASE+'/'+rel,sha256='a'*64,
            checksum_path='data/raw/'+rel+'.CHECKSUM',checksum_sha256='b'*64))
    assert len(data.validate_archive_manifest({'archives':entries}))==107
    with pytest.raises(ValueError,match='exact'):
        data.validate_archive_manifest({'archives':entries[:-1]})
    escaped=[dict(x) for x in entries];escaped[0]['path']='../outside.zip'
    with pytest.raises(ValueError,match='path'):
        data.validate_archive_manifest({'archives':escaped})


@pytest.mark.parametrize('btc,post',[(False,False),(False,True),(True,True)])
def test_load_gate_precedes_all_file_access(monkeypatch,btc,post):
    monkeypatch.setattr(scope,'BTC_IS_UNLOCKED',btc)
    monkeypatch.setattr(scope,'POST_IS_AND_ETH_UNLOCKED',post)
    def forbidden(*args,**kwargs):raise AssertionError('loader touched disk before authorization')
    monkeypatch.setattr(data.Path,'read_text',forbidden)
    monkeypatch.setattr(data.Path,'read_bytes',forbidden)
    with pytest.raises(PermissionError):data.load_inputs('/synthetic/no_market')


def canonical_funding_fixture():
    launch=pd.DatetimeIndex(['2020-08-11T16:00Z','2020-08-12T00:00Z']).as_unit('ns')
    common=pd.date_range('2020-09-01','2024-01-01',freq='8h',inclusive='left',tz='UTC').as_unit('ns')
    index=launch.append(common)
    return pd.Series(np.resize([-.0001,.0002],len(index)),index=index,name='funding_rate')


def test_approved_archive_set_has_only_eighteen_vision_funding_months():
    entries=data.expected_archives()
    funding=[entry for entry in entries if entry[1]=='fundingRate']
    assert len(funding)==18
    assert {entry[2] for entry in funding}==set(pd.period_range('2022-07','2023-12',freq='M').astype(str))
    assert sum(entry[0]=='SPOT' and entry[1]=='klines' for entry in entries)==48
    assert sum(entry[0]=='CM' and entry[1]=='klines' for entry in entries)==41


def test_canonical_funding_requires_complete_common_calendar_but_preserves_launch_records():
    source=canonical_funding_fixture()
    native=pd.Timestamp('2020-08-11T15:00Z')
    checked=data.validate_cm_funding(source,native)
    pd.testing.assert_series_equal(checked,source,check_exact=True)
    assert (checked.index>=data.CM_COMMON_START).sum()==3651
    assert checked.index[0]==pd.Timestamp('2020-08-11T16:00Z')
    assert len(checked)==3653  # No invented earlier or missing launch-month settlements.
    with pytest.raises(ValueError,match='3651'):
        data.validate_cm_funding(source.drop(pd.Timestamp('2020-09-01T00:00Z')),native)
    with pytest.raises(ValueError,match='unique'):
        data.validate_cm_funding(pd.concat([source.iloc[:1],source]),native)
    with pytest.raises(ValueError,match='native IS'):
        data.validate_cm_funding(source,pd.Timestamp('2020-08-12T00:00Z'))


def test_canonical_funding_rejects_future_nonfinite_and_off_grid_values():
    source=canonical_funding_fixture();native=pd.Timestamp('2020-08-11T15:00Z')
    future=pd.concat([source,pd.Series([.001],index=pd.DatetimeIndex(['2024-01-01T00:00Z']).as_unit('ns'))])
    with pytest.raises(ValueError,match='native IS'):data.validate_cm_funding(future,native)
    nonfinite=source.copy();nonfinite.iloc[5]=np.nan
    with pytest.raises(ValueError,match='finite'):data.validate_cm_funding(nonfinite,native)
    off_grid=source.copy();times=off_grid.index.to_list();times[0]+=pd.Timedelta(minutes=1)
    off_grid.index=pd.DatetimeIndex(times).as_unit('ns')
    with pytest.raises(ValueError,match='8-hour'):data.validate_cm_funding(off_grid,native)


def test_loader_delegates_registered_payload_verification_before_canonical_use(monkeypatch):
    from trend_hold_venue_study import funding_sources
    monkeypatch.setattr(scope,'BTC_IS_UNLOCKED',True)
    monkeypatch.setattr(scope,'POST_IS_AND_ETH_UNLOCKED',False)
    source=canonical_funding_fixture();seen=[]
    def verified(root):
        seen.append(root)
        return source
    monkeypatch.setattr(funding_sources,'verify_registered',verified)
    result=data.load_cm_funding('/synthetic/root',pd.Timestamp('2020-08-11T15:00Z'))
    assert seen==[data.Path('/synthetic/root')]
    pd.testing.assert_series_equal(result,source,check_exact=True)
    def bad_payload(root):raise ValueError('registered raw API hash mismatch')
    monkeypatch.setattr(funding_sources,'verify_registered',bad_payload)
    with pytest.raises(ValueError,match='API hash'):
        data.load_cm_funding('/synthetic/root',pd.Timestamp('2020-08-11T15:00Z'))

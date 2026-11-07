"""Scheme A funding validation with supplied synthetic records only."""
from decimal import Decimal
import json
import pytest
from trend_hold_venue_study import funding_sources as f,scope


def api_row(stamp,rate='0.00010000'):
    return {'symbol':'BTCUSD_PERP','fundingTime':stamp,'fundingRate':rate}


def test_minute_alignment_preserves_r5_rounding_and_8h_grid():
    t=f.OVERLAP_START
    assert f.align_time(t+999)==(t,999)
    assert f.align_time(t-999)==(t,-999)
    for delta in (60000,28800000//2,59000):
        with pytest.raises(ValueError):f.align_time(t+delta)


def test_decimal_tail_policy_and_missing_duplicate_failures():
    t=f.OVERLAP_START
    api=f.parse_api(json.dumps([api_row(t),api_row(t+f.STEP)]).encode(),t,t+2*f.STEP-1)
    vision=[{**api[0],'rate':'0.000100000000000000'}, {**api[1],'rate':'0.00010001'}]
    report=f.compare_overlap(api,vision,expected_times=[t,t+f.STEP])
    assert report['exact_decimal_match_count']==1
    assert Decimal(report['max_abs_decimal_diff'])==Decimal('1e-8')
    assert report['within_authorized_tail_tolerance_count']==1
    with pytest.raises(ValueError):f.compare_overlap(api,[vision[0]],expected_times=[t,t+f.STEP])
    with pytest.raises(ValueError):f.compare_overlap(api,[vision[0],vision[0]],expected_times=[t,t+f.STEP])
    with pytest.raises(ValueError):f.compare_overlap(api,[vision[0],{**vision[1],'rate':'0.00010001000001'}],expected_times=[t,t+f.STEP])


def test_api_scope_nonfinite_and_raw_request_bounds():
    t=f.OVERLAP_START
    for row in (api_row(t,'NaN'),{**api_row(t),'symbol':'ETHUSD_PERP'},api_row(t-1),api_row(t+f.STEP)):
        with pytest.raises(ValueError):f.parse_api(json.dumps([row]).encode(),t,t+f.STEP-1)
    with pytest.raises(ValueError):f.parse_api(b'{}',t,t+f.STEP-1)
    with pytest.raises(ValueError):f.validate_request({'symbol':'BTCUSD_PERP','startTime':f.END,'endTime':f.END+1,'limit':1000})


def test_vision_decimal_precision_and_interval_checks():
    t=f.OVERLAP_START
    rows=f.parse_vision(f'calc_time,funding_interval_hours,last_funding_rate\n{t+2},8,0.000100000000000001\n'.encode())
    assert rows[0]['rate']=='0.000100000000000001'
    assert rows[0]['time']==t
    with pytest.raises(ValueError):f.parse_vision(f'calc_time,funding_interval_hours,last_funding_rate\n{t},4,0.0001\n'.encode())


def test_fixed_authority_splice_complete_grid_and_august_registration():
    times=list(range(f.NATIVE_LOWER,f.END,f.STEP))
    api=[{'time':t,'rate':'0.0001','raw_time':t,'offset_ms':0} for t in times]
    vision=[{**r,'rate':'0.000100001'} for r in api if r['time']>=f.OVERLAP_START and r['time'] not in set(f.approved_month_end_times())]
    result=f.assemble(api,vision)
    assert result['common_settlements']==3651
    assert result['overlap']['settlements']==1593
    assert result['supplemented_overlap_settlements']==1647
    assert result['month_end_supplement_count']==54
    assert len({r['funding_time_ms'] for r in result['month_end_supplements']})==54
    assert result['canonical_records'][0]['source']=='dapi'
    assert result['canonical_records'][-1]['source']=='dapi_month_end_supplement'
    assert result['canonical_records'][-1]['funding_rate']=='0.0001'
    assert result['canonical_records'][-4]['source']=='vision'
    assert result['canonical_records'][-4]['funding_rate']=='0.000100001'
    assert result['august_launch']['continuous_from_first_settlement']
    with pytest.raises(ValueError):f.assemble(api[1:2]+api[3:],vision)
    with pytest.raises(ValueError):f.assemble(api+[api[-1]],vision)
    with pytest.raises(ValueError,match='exact 54'):f.assemble(api,vision[1:])
    with pytest.raises(ValueError):f.assemble(api,vision+[vision[-1]])
    with pytest.raises(ValueError,match='Decimal difference'):f.assemble(api,[{**vision[0],'rate':'0.001'}]+vision[1:])


def test_preparation_requires_both_locks_false_before_filesystem(tmp_path,monkeypatch):
    monkeypatch.setattr(scope,'BTC_IS_UNLOCKED',True)
    with pytest.raises(PermissionError):f.prepare(tmp_path)
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(scope,'BTC_IS_UNLOCKED',False)
    monkeypatch.setattr(scope,'POST_IS_AND_ETH_UNLOCKED',True)
    with pytest.raises(PermissionError):f.prepare(tmp_path)


def test_saved_resume_is_locked_before_filesystem_and_has_no_network(tmp_path,monkeypatch):
    monkeypatch.setattr(scope,'BTC_IS_UNLOCKED',True)
    with pytest.raises(PermissionError):f.register_saved(tmp_path)
    monkeypatch.setattr(scope,'BTC_IS_UNLOCKED',False)
    monkeypatch.setattr(scope,'POST_IS_AND_ETH_UNLOCKED',True)
    with pytest.raises(PermissionError):f.register_saved(tmp_path)
    assert not list(tmp_path.iterdir())


def test_only_54_preapproved_month_end_timestamps_are_eligible():
    from datetime import datetime,timezone
    import calendar
    times=f.approved_month_end_times()
    assert len(times)==len(set(times))==54
    for t in times:
        d=datetime.fromtimestamp(t/1000,timezone.utc)
        assert d.day==calendar.monthrange(d.year,d.month)[1]
        assert d.hour in (0,8,16) and d.minute==0
        assert f.OVERLAP_START<=t<f.END

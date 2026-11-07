"""Synthetic-only orchestration/export checks; no historical inputs are opened."""
import csv
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trend_hold_venue_study import runtime, scope
from trend_hold_venue_study.models import Result


def tiny_result():
    days = pd.date_range('2020-03-01', periods=2, freq='D', tz='UTC').as_unit('ns')
    minute = pd.DataFrame({'equity':[100001.,100002.], 'bad':False},
                          index=days + pd.Timedelta(hours=23,minutes=59))
    equity = pd.Series([100001.,100002.],index=days,name='equity')
    returns = pd.Series([100001./100000.-1.,100002./100001.-1.],index=days,name='return')
    attribution = pd.DataFrame({'time':minute.index,
        'amount':np.array([np.longdouble('1.234567890123456789'),np.longdouble('-0.000000000000000000123')])})
    return Result(minute,equity,returns,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),
        {'version':'V1_USDM','cost':'gross_0','initial_capital':100000.,'final_equity':100002.,
         'order_count':2,'price_pnl':2.,'total_funding':0.,'total_fees':0.,'total_slippage_cost':0.,
         'account_reconciliation_error':0.},attribution)


def test_registered_matrix_is_exactly_42_unique_jobs_in_declared_order():
    rows=runtime.jobs()
    assert len(rows)==42 and len({r['job_id'] for r in rows})==42
    assert [r['window'] for r in rows[:12]]==['IS-Full']*12
    assert [r['window'] for r in rows[12:36]]==['IS-Common']*24
    assert all(r['whole_contracts'] for r in rows[36:39])
    assert [r['version'] for r in rows[36:39]]==['V3_CM_DeltaMatched','P_CM_DeltaMatched','V4_CM_CoinAccum']
    assert all(r['signal_source']=='CM' and r['window']=='IS-CM-SelfSignal' for r in rows[39:])
    assert all(r['cost']=='main_17' for r in rows[36:])
    assert [r['cost'] for r in rows[:3]]==['gross_0','main_17','harsh_34']
    assert len(runtime.PAIRED_FAMILIES)==5
    assert not any('ETH' in str(r) or r['window']=='Post-IS' for r in rows)


def test_r5_equality_detects_single_ulp_day_and_terminal_differences():
    result=tiny_result()
    daily=pd.DataFrame({'equity':result.daily_equity,'return':result.daily_returns})
    metadata={'final_equity':result.metadata['final_equity']}
    assert runtime.assert_r5_equal(result,daily,metadata)['daily_exact']
    altered=daily.copy();altered.iloc[0,0]=np.nextafter(altered.iloc[0,0],np.inf)
    with pytest.raises(ValueError,match='R5.*daily'):
        runtime.assert_r5_equal(result,altered,metadata)
    with pytest.raises(ValueError,match='R5.*terminal'):
        runtime.assert_r5_equal(result,daily,{'final_equity':np.nextafter(100002.,np.inf)})
    wrong=daily.copy();wrong.index=wrong.index+pd.Timedelta(days=1)
    with pytest.raises(ValueError,match='R5.*calendar'):
        runtime.assert_r5_equal(result,wrong,metadata)


def test_exports_keep_extended_precision_attribution_and_daily_columns(tmp_path):
    result=tiny_result()
    account=runtime.export_account(result,tmp_path/'account',runtime.jobs()[0])
    loaded=pd.read_parquet(tmp_path/'account/minute.parquet')
    pd.testing.assert_frame_equal(loaded,result.minute)
    import gzip
    with gzip.open(tmp_path/'account/attribution.csv.gz','rt',newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert np.longdouble(rows[0]['amount'])==result.attribution['amount'].iloc[0]
    assert np.longdouble(rows[1]['amount'])==result.attribution['amount'].iloc[1]
    assert list(account['daily'])==['equity','return']
    assert account['job']['job_id']==runtime.jobs()[0]['job_id']
    assert 'minute' not in account
    assert (tmp_path/'account/pressure_events.csv.gz').exists()


def test_execution_records_first_error_and_never_retries_or_overwrites(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(runtime,'_validate_inputs',lambda contexts:'synthetic')
    days=pd.date_range('2020-03-01',periods=2,tz='UTC').as_unit('ns')
    feature=pd.DataFrame({'g':[True,False],'budget':[.5,.5]},index=days)
    monkeypatch.setattr(runtime,'_prepare_features',lambda contexts:{'UM':feature,'CM':feature.copy()})
    def fail(*args):
        calls.append(args[2]['job_id'])
        error=ValueError('synthetic first failure')
        error.partial={'orders':pd.DataFrame({'qty':[1.]}),'reason':'synthetic'}
        raise error
    monkeypatch.setattr(runtime,'_run_account',fail)
    output=tmp_path/'single_attempt'
    with pytest.raises(ValueError,match='synthetic first failure'):
        runtime.execute({},output,{'synthetic':True})
    assert calls==[runtime.jobs()[0]['job_id']]
    stopped=json.loads((output/'STOPPED.json').read_text())
    assert stopped['completed_jobs']==[]
    assert stopped['current_job']==runtime.jobs()[0]['job_id']
    assert not stopped['automatic_retry']
    assert (output/'STOPPED_partial_orders.csv.gz').exists()
    with pytest.raises(FileExistsError):runtime.execute({},output,{'synthetic':True})
    assert len(calls)==1


def test_runtime_rejects_post_is_eth_and_real_locked_before_feature_work(monkeypatch):
    frames={v:SimpleNamespace(source='real',asset='BTC',venue=v) for v in ('UM','SPOT','CM')}
    with pytest.raises(scope.RealDataLockedError):runtime._validate_inputs(frames)
    synthetic={v:SimpleNamespace(source='synthetic',asset='ETH',venue=v) for v in ('UM','SPOT','CM')}
    with pytest.raises(ValueError,match='BTC'):
        runtime._validate_inputs(synthetic)
    monkeypatch.setattr(scope,'POST_IS_AND_ETH_UNLOCKED',True)
    with pytest.raises(ValueError,match='POST'):
        runtime._validate_inputs({})


def test_signed_cross_venue_bridge_reconciles_without_claiming_causal_basis():
    records=[]
    for version,equity,funding,fee,slip,conversion,basis in [
        ('V1_USDM',100010.,-2.,1.,-1.,0.,0.),
        ('V2_SPOT',100014.,0.,2.,-1.,0.,0.),
        ('V3_CM_DeltaMatched',100017.,3.,1.,-1.,-2.,4.),
    ]:
        records.append(dict(window='IS-Common',version=version,cost='main_17',whole_contracts=False,
            signal_source='UM',final_equity=equity,funding_usd=funding,fee_usd=fee,
            slippage_usd=slip,conversion_usd=conversion,basis_final_usd=basis,
            cash_clip_count=1 if version=='V2_SPOT' else 0,
            zero_floor_adjustment_usd=0.,cash_roundoff_adjustment_usd=0.))
    bridge=runtime.crossvenue_differences(pd.DataFrame(records))
    assert len(bridge)==2
    for row in bridge.to_dict('records'):
        components=(row['funding_difference_usd']+row['fee_pnl_difference_usd']+
            row['slippage_difference_usd']+row['conversion_difference_usd']+
            row['basis_endpoint_difference_usd']+row['zero_floor_difference_usd']+
            row['cash_roundoff_difference_usd']+row['price_and_path_residual_usd'])
        assert components==row['equity_difference_usd']
        assert abs(row['bridge_error_usd'])<1e-9


def test_zero_floor_and_cash_roundoff_are_not_hidden_in_price_path_residual():
    common=dict(window='IS-Common',cost='main_17',whole_contracts=False,signal_source='UM',
        funding_usd=0.,fee_usd=0.,slippage_usd=0.,conversion_usd=0.,basis_final_usd=0.,
        cash_clip_count=0,zero_floor_adjustment_usd=0.,cash_roundoff_adjustment_usd=0.)
    roundoff=np.spacing(100000.)  # Exactly representable positive clamp below CASH_EPS.
    rows=[dict(common,version='V1_USDM',final_equity=100000.),
          dict(common,version='V2_SPOT',final_equity=100000.+roundoff,
               cash_roundoff_adjustment_usd=roundoff),
          dict(common,version='V3_CM_DeltaMatched',final_equity=100007.,
               zero_floor_adjustment_usd=7.)]
    bridge=runtime.crossvenue_differences(pd.DataFrame(rows)).set_index('comparison')
    assert bridge.loc['V2_SPOT','cash_roundoff_difference_usd']==roundoff
    assert bridge.loc['V3_CM_DeltaMatched','zero_floor_difference_usd']==7.
    assert (bridge['price_and_path_residual_usd']==0.).all()
    assert (bridge['bridge_error_usd']==0.).all()

def test_comparison_exports_existing_spot_roundoff_and_coin_floor_terms():
    result=tiny_result()
    result.attribution['cash_roundoff_adjustment']=np.array([np.spacing(100000.),0.],dtype=np.longdouble)
    result.attribution['zero_floor_spot_usd']=np.array([0.,7.],dtype=np.longdouble)
    row=runtime.comparison_row(result,runtime.jobs()[0])
    assert row['cash_roundoff_adjustment_usd']==np.spacing(100000.)
    assert row['zero_floor_adjustment_usd']==7.

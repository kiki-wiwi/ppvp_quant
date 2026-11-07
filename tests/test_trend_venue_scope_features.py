"""No external input: causal features and independent scope barriers."""
import numpy as np
import pandas as pd
import pytest
from trend_hold_venue_study import scope, features
from trend_hold_venue_study.config import window
from trend_hold_study.scope import make_context as old_context
from trend_hold_study.features import build_features as old_features

def fixture(start='2020-01-01',days=70,asset='BTC',venue='UM'):
    idx=pd.date_range(pd.Timestamp(start,tz='UTC'),periods=days*1440,freq='min')
    daily=100+np.arange(days)*.3+np.sin(np.arange(days))*2
    p=np.repeat(daily,1440)
    frame=pd.DataFrame(dict(open=p,high=p+1,low=p-1,close=p,volume=1.),index=idx)
    frame.attrs['provenance']='synthetic'
    return frame,scope.make_context(frame,asset=asset,venue=venue)

def test_two_real_locks_and_provenance(monkeypatch):
    assert not scope.BTC_IS_UNLOCKED and not scope.POST_IS_AND_ETH_UNLOCKED
    for asset,venue,start in [('BTC','UM','2020-01-01'),('BTC','CM','2024-01-01'),('ETH','SPOT','2020-07-01')]:
        with pytest.raises(scope.RealDataLockedError):
            scope.require_real_authorization(asset,venue,start+'T00:00Z', '2024-02-01T00:00Z')
    monkeypatch.setattr(scope,'BTC_IS_UNLOCKED',True)
    scope.require_real_authorization('BTC','UM','2020-01-01T00:00Z','2024-01-01T00:00Z')
    for asset,start in [('BTC','2024-01-01'),('ETH','2020-07-01')]:
        with pytest.raises(scope.RealDataLockedError):
            scope.require_real_authorization(asset,'UM',start+'T00:00Z','2024-02-01T00:00Z')
    frame,_=fixture(days=1)
    frame.attrs['source']='real'
    with pytest.raises(scope.ScopeError): scope.make_context(frame)

def test_immutable_and_utc_ns():
    frame,ctx=fixture(days=1)
    frame.iloc[0,0]=999
    assert ctx.O[0]!=999
    assert ctx.index.unit=='ns'
    with pytest.raises(ValueError):ctx.C.setflags(write=True)
    with pytest.raises(ValueError):ctx.index.asi8.setflags(write=True)
    _,future=fixture(start='2025-06-30',days=2,asset='ETH',venue='CM')
    assert future.asset=='ETH'

def test_r5_features_exact_and_prefix_causal():
    frame,ctx=fixture()
    origin=pd.Timestamp('2020-03-01',tz='UTC')
    got=features.build_features(ctx,origin)
    expected=old_features(old_context(frame))
    pd.testing.assert_frame_equal(got,expected,check_exact=True)
    cut=frame.loc[:'2020-03-05 23:59'].copy()
    trunc=features.build_features(scope.make_context(cut),origin)
    pd.testing.assert_frame_equal(got.loc[trunc.index],trunc,check_exact=True)
    changed=frame.copy();changed.loc['2020-03-06':,['open','high','low','close']]*=2
    after=features.build_features(scope.make_context(changed),origin)
    pd.testing.assert_frame_equal(got.loc[:'2020-03-06'],after.loc[:'2020-03-06'],check_exact=True)

def test_common_budget_inherits_monday_and_flat_is_off():
    frame,ctx=fixture(start='2020-07-01',days=70,asset='ETH')
    schedule=features.build_features(ctx,window('IS-Common','ETH').signal_origin)
    mon=pd.Timestamp('2020-08-31',tz='UTC'); tue=mon+pd.Timedelta(days=1)
    assert schedule.loc[tue,'budget']==schedule.loc[mon,'budget']
    assert not schedule.loc[tue,'budget_update']
    frame.loc[:,['open','high','low','close']]=100.
    flat=features.build_features(scope.make_context(frame,asset='ETH'),mon)
    assert not flat.g.any() and (flat.budget==1).all()

def test_last_valid_daily_and_empty_day_stop():
    frame,ctx=fixture()
    frame['bad']=False
    frame.loc['2020-02-29 23:59','bad']=True
    frame.loc['2020-02-29 23:59','close']=100000.
    frame.loc['2020-02-29 23:59','high']=100001.
    ctx=scope.make_context(frame)
    refs=features.references(ctx,pd.DatetimeIndex([pd.Timestamp('2020-03-01',tz='UTC')]))
    assert refs.iloc[0].eref==frame.loc['2020-02-29 23:58','close']
    frame.loc['2020-02-29','bad']=True
    with pytest.raises(ValueError,match='no valid'):
        features.build_features(scope.make_context(frame),pd.Timestamp('2020-03-01',tz='UTC'))

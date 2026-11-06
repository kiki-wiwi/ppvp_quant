"""Synthetic-only boundary and provenance checks for the independent study."""
import numpy as np
import pandas as pd
import pytest


def frame():
    idx=pd.date_range('2020-03-01',periods=3,freq='min',tz='UTC')
    f=pd.DataFrame({'open':100.,'high':101.,'low':99.,'close':100.,'volume':5.},index=idx)
    f.attrs['provenance']='synthetic'
    return f


def test_independent_real_lock_rejects_before_frame_access():
    from dip_limit_study import scope
    assert scope.REAL_DATA_UNLOCKED is False
    with pytest.raises(scope.RealDataLockedError):
        scope.make_context(None,source='real')


def test_synthetic_snapshot_cannot_be_mutated_or_relabelled():
    from dip_limit_study import scope
    f=frame(); c=scope.make_context(f); f.iloc[0,0]=88
    assert c.O[0]==100
    with pytest.raises(ValueError): c.O.setflags(write=True)
    f.attrs['source']='real'
    with pytest.raises(scope.ScopeError): scope.make_context(f)


def test_only_btc_is_continuous_timezone_aware_context_is_accepted():
    from dip_limit_study import scope
    with pytest.raises(scope.ScopeError): scope.make_context(frame(),symbol='ETHUSDT')
    for bad in ('2024-01-01','2019-12-31'):
        f=frame(); f.index=pd.date_range(bad,periods=3,freq='min',tz='UTC')
        with pytest.raises(scope.ScopeError): scope.make_context(f)
    f=frame().iloc[[0,2]]
    with pytest.raises(scope.ScopeError): scope.make_context(f)


def test_utc_normalization_and_funding_boundary_positions():
    from dip_limit_study import scope
    f=frame(); f.index=f.index.tz_convert('Asia/Shanghai').as_unit('us')
    c=scope.make_context(f,pd.Series([.001],index=f.index[:1]))
    assert str(c.index.tz)=='UTC' and c.index.unit=='ns'
    assert c.funding_rate[0]==.001 and np.isnan(c.funding_rate[1])

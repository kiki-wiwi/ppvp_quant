"""Synthetic authorization and immutable independent context boundary."""
import numpy as np
import pandas as pd
import pytest


def fixture_frame():
    x = pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=1.),
        index=pd.date_range('2020-03-01', periods=120, freq='min', tz='UTC'))
    x.attrs['provenance'] = 'synthetic'
    return x


def test_new_lock_is_closed_independently_of_old_study():
    from trend_hold_study import scope
    if scope.REAL_DATA_UNLOCKED:
        pytest.skip('source-lock-only assertion applies before approved unlock')
    assert scope.REAL_DATA_UNLOCKED is False
    with pytest.raises(scope.RealDataLockedError):
        scope.make_context(None, source='real')


def test_snapshots_cannot_be_rewritten_or_relabelled():
    from trend_hold_study import scope
    frame = fixture_frame()
    ctx = scope.make_context(frame)
    frame.iloc[0,0] = 200.
    assert ctx.O[0] == 100
    with pytest.raises(ValueError): ctx.O.setflags(write=True)
    with pytest.raises(ValueError): ctx.index.asi8.setflags(write=True)
    frame.attrs['source'] = 'real'
    with pytest.raises(scope.ScopeError): scope.make_context(frame)


def test_symbol_and_window_and_units_are_enforced():
    from trend_hold_study import scope
    frame = fixture_frame()
    with pytest.raises(scope.ScopeError): scope.make_context(frame, symbol='ETHUSDT')
    frame.index = frame.index.tz_convert('Asia/Tokyo').as_unit('us')
    assert scope.make_context(frame).index.unit == 'ns'
    frame.index = frame.index + pd.Timedelta(days=1500)
    with pytest.raises(scope.ScopeError): scope.make_context(frame)


def test_noncontinuous_and_invalid_price_grids_fail():
    from trend_hold_study import scope
    frame = fixture_frame().drop(fixture_frame().index[1])
    frame.attrs['provenance'] = 'synthetic'
    with pytest.raises(scope.ScopeError): scope.make_context(frame)
    frame = fixture_frame(); frame.iloc[0,0] = np.nan
    with pytest.raises(scope.ScopeError): scope.make_context(frame)

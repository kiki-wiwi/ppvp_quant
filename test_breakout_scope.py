"""Synthetic-only scope, immutability and real-data lock checks."""
import numpy as np
import pandas as pd
import pytest


def frame(n=120, start='2021-01-01'):
    idx = pd.date_range(start, periods=n, freq='min', tz='UTC')
    f = pd.DataFrame(dict(open=100., high=101., low=99., close=100.,
                          volume=10., bad=False, archive_gap=False), index=idx)
    f.attrs['provenance']='synthetic'
    return f


def test_context_is_an_independent_readonly_snapshot():
    from breakout_study.scope import make_context
    f = frame()
    ctx = make_context(f)
    f.iloc[0, 0] = 500
    assert ctx.O[0] == 100
    with pytest.raises(ValueError):
        ctx.C[0] = 999
    with pytest.raises(ValueError):
        ctx.C.setflags(write=True)
    assert str(ctx.index.dtype) == 'datetime64[ns, UTC]'


@pytest.mark.parametrize('kind', ['naive','duplicate','gap','not_minute','val','eth','real','nan','broken_ohlc'])
def test_invalid_context_is_rejected_before_matching(kind):
    from breakout_study.scope import make_context, ScopeError, RealDataLockedError
    f=frame(); kw={}
    if kind=='naive': f.index=f.index.tz_localize(None)
    elif kind=='duplicate': f.index=pd.DatetimeIndex([f.index[0]]*len(f))
    elif kind=='gap': f=f.drop(f.index[5])
    elif kind=='not_minute': f.index=f.index+pd.Timedelta(seconds=1)
    elif kind=='val': f.index=f.index+pd.DateOffset(years=3)
    elif kind=='eth': kw['symbol']='ETHUSDT'
    elif kind=='real': kw['source']='real'
    elif kind=='nan': f.iloc[0,0]=np.nan
    elif kind=='broken_ohlc': f.iloc[0,0]=102.
    with pytest.raises((ScopeError, RealDataLockedError)):
        make_context(f,**kw)


def test_funding_rounds_to_minute_and_rejects_duplicate_or_outside_grid():
    from breakout_study.scope import make_context, ScopeError
    f=frame()
    funding=pd.Series([.001], index=[f.index[10]+pd.Timedelta(milliseconds=2)])
    ctx=make_context(f, funding=funding)
    assert ctx.funding_rate[10]==.001
    assert np.isnan(ctx.funding_rate[9])
    for bad in (pd.concat([funding,funding]),pd.Series([.1],index=[f.index[-1]+pd.Timedelta(minutes=1)])):
        with pytest.raises(ScopeError): make_context(f,funding=bad)


def test_real_entry_denies_before_io(monkeypatch):
    from breakout_study import data
    def fail(*a,**kw): raise AssertionError('market I/O attempted')
    monkeypatch.setattr(pd,'read_parquet',fail)
    with pytest.raises(data.RealDataLockedError): data.load_real_context()


def test_untagged_frames_cannot_be_accidentally_matched_as_synthetic():
    from breakout_study.scope import make_context, ScopeError
    f=frame();f.attrs.clear()
    with pytest.raises(ScopeError): make_context(f)


def test_approved_costs_and_no_cli_unlock():
    from breakout_study.config import COSTS, VERSIONS
    assert tuple(COSTS)==('gross_0','main_17','harsh_34')
    assert VERSIONS==('P_channel','V_vah')
    assert COSTS['main_17'].c_rt==pytest.approx(.0017)
    assert COSTS['harsh_34'].c_rt==pytest.approx(.0034)
    with pytest.raises(TypeError): COSTS['extra']=COSTS['gross_0']

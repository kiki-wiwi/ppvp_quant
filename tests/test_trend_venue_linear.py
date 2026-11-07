"""Synthetic-only exact regression against frozen R5 linear bookkeeping."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from trend_hold_venue_study.config import COSTS, Window
from trend_hold_venue_study.linear import run_linear
from trend_hold_venue_study.scope import Context


def fixture(*, start='2020-03-01', asset='BTC', trend=(True,), budget=None,
            opens=None, closes=None, bad=(), gap=(), rates=None):
    start = pd.Timestamp(start, tz='UTC')
    index = pd.date_range(start-pd.Timedelta(minutes=1),
                          periods=len(trend)*1440+1, freq='min').as_unit('ns')
    op = np.full(len(index), 100.)
    cl = op.copy()
    for m, p in (opens or {}).items(): op[m+1] = p
    for m, p in (closes or {}).items(): cl[m+1] = p
    bd = np.zeros(len(index), bool)
    gp = bd.copy()
    bd[np.asarray(bad, dtype=int)+1] = True
    gp[np.asarray(gap, dtype=int)+1] = True
    fr = np.full(len(index), np.nan)
    for m, r in (rates or {}).items(): fr[m+1] = r
    ctx = Context(index, op, np.maximum(op, cl), np.minimum(op, cl), cl,
                  np.ones(len(index)), bd, gp, fr, asset=asset, venue='UM')
    days = index[1::1440]
    features = pd.DataFrame({
        'eref': cl[np.arange(len(days))*1440], 'sma60': 100., 'sigma': .4,
        'budget': .5 if budget is None else budget, 'g': trend,
        'budget_update': (days == start) | (days.dayofweek == 0),
        'n_valid_prior': 1440, 'has_archive_gap': False,
    }, index=days)
    window = Window('synthetic', start, start+pd.Timedelta(days=len(trend)), start)
    return ctx, features, window


@pytest.mark.parametrize('cost', tuple(COSTS))
@pytest.mark.parametrize('version,old_version', [('V1_USDM','T_daily_trend'),
                                               ('P_USDM','P_continuous_long')])
@pytest.mark.parametrize('kwargs', [
    dict(trend=(True, True), budget=(.5,.25),
         opens={0:110.,1440:120.}, rates={0:.5,1440:.01}),
    dict(trend=(True,True,True), budget=(.5,.25,.25), bad=(1440,), gap=(1440,),
         opens={1440:900.}, closes={1440:999.,2879:125.}, rates={1440:.01}),
    dict(trend=(True,False,True,True), bad=(1440,),
         opens={5759:85.}, closes={5759:130.}, rates={960:-.01}),
])
def test_E_exact_minute_and_ledger_regression(kwargs, version, old_version, cost):
    from trend_hold_study.engine import run_portfolio
    from trend_hold_study.scope import Context as FrozenContext
    ctx, features, window = fixture(**kwargs)
    frozen_ctx = FrozenContext(ctx.index, ctx.O, ctx.H, ctx.L, ctx.C, ctx.V,
                               ctx.bad, ctx.archive_gap, ctx.funding_rate)
    old = run_portfolio(frozen_ctx, features, old_version, COSTS[cost])
    new = run_linear(ctx, features, version, COSTS[cost], window)
    for field in ('minute','orders','decisions','funding'):
        assert_frame_equal(getattr(new,field),getattr(old,field),check_exact=True)
    assert_series_equal(new.daily_equity, old.daily_equity, check_exact=True)
    assert_series_equal(new.daily_returns, old.daily_returns, check_exact=True)
    for key,value in old.metadata.items():
        if key != 'version': assert new.metadata[key] == value


def test_synthetic_eth_future_calendar_and_terminal_close():
    ctx, features, window = fixture(start='2024-01-01',asset='ETH',
        trend=(True,False,True), rates={0:.2,1440:.01}, closes={4319:130.})
    result = run_linear(ctx,features,'V1_USDM',COSTS['main_17'],window)
    assert result.decisions.g.tolist() == [True,False,True]
    assert result.decisions.budget_update.tolist() == [True,False,False]
    assert result.funding.qty_before.tolist() == [0.,500.]
    assert result.funding.cash_flow.tolist() == [0.,-500.]
    assert result.orders.iloc[-1].reference_price == 130.
    assert result.orders.iloc[-1].time == window.end-pd.Timedelta(nanoseconds=1)
    assert result.minute.qty.iloc[-1] == 0.


def test_inherited_tuesday_budget_is_not_reestimated():
    ctx,features,window = fixture(start='2020-09-01',trend=(True,True))
    window = Window('synthetic',window.start,window.end,
                    pd.Timestamp('2020-08-31',tz='UTC'))
    features['budget_update'] = False
    result = run_linear(ctx,features,'V1_USDM',COSTS['gross_0'],window)
    assert result.decisions.initial.tolist() == [True,False]
    assert result.decisions.triggered.tolist() == [True,False]
    assert result.orders.qty_target.tolist() == [500.,0.]


def test_context_tail_is_not_traded_or_funded():
    ctx,features,full = fixture(trend=(True,False), rates={1440:.9})
    short = Window('synthetic',full.start,full.start+pd.Timedelta(days=1),full.start)
    result = run_linear(ctx,features.iloc[:1],'V1_USDM',COSTS['main_17'],short)
    assert len(result.minute) == 1440
    assert result.funding.empty
    assert result.orders.time.iloc[-1] == short.end-pd.Timedelta(nanoseconds=1)


def test_rejects_invalid_window_reference_and_preserves_failure_state():
    ctx,features,window = fixture()
    invalid = features.copy()
    invalid['eref'] = 101.
    with pytest.raises(ValueError,match='reference'):
        run_linear(ctx,invalid,'V1_USDM',COSTS['gross_0'],window)
    ctx,features,window = fixture(budget=(1.,),rates={480:1.})
    with pytest.raises(ValueError,match='equity') as stopped:
        run_linear(ctx,features,'V1_USDM',COSTS['main_17'],window)
    assert stopped.value.partial['failure_time'] == ctx.index[481]
    assert stopped.value.partial['orders'].delta_qty.tolist() == [1000.]
    assert stopped.value.partial['funding'].cash_flow.tolist() == [-100000.]

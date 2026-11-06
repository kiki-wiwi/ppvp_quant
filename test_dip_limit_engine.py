"""Hand-calculated synthetic execution checks; never load registered market data."""
import importlib

import numpy as np
import pandas as pd
import pytest


START = pd.Timestamp('2020-03-01 01:00', tz='UTC')


def context(n=1502, *, start=START, price=100., funding=None, changes=None, bad=()):
    from dip_limit_study.scope import make_context
    ix = pd.date_range(start, periods=n, freq='min')
    frame = pd.DataFrame({'open': price, 'high': price + .2, 'low': price - .2,
                          'close': price, 'volume': 1000., 'bad': False,
                          'archive_gap': False}, index=ix)
    for i, values in (changes or {}).items():
        for k, v in values.items():
            frame.loc[ix[i], k] = v
    for i in bad:
        frame.loc[ix[i], 'bad'] = True
    frame.attrs['provenance'] = 'synthetic'
    return make_context(frame, funding=funding)


def signals(*offsets, eref=100., atr=5., target=110.):
    from dip_limit_study.features import floor_price_offset
    offsets = offsets or (0,)
    ix = pd.DatetimeIndex([START + pd.Timedelta(minutes=x) for x in offsets])
    return pd.DataFrame({'signal_time': ix - pd.Timedelta(hours=1),
                         'decision_time': ix,
                         'deadline': ix + pd.Timedelta(hours=24),
                         'eref': eref, 'atr': atr, 'sl': floor_price_offset(eref, atr, 2.),
                         'limit': floor_price_offset(eref, atr, .5), 'target': target,
                         'eligible': True, 'skip_reason': '',
                         'signal_lookback_has_archive_gap': False}, index=ix)


def run(ctx, sig=None, version='M_market', cost='gross', **kw):
    return importlib.import_module('dip_limit_study.engine').run_portfolio(
        ctx, signals() if sig is None else sig, version, cost, **kw)


def test_market_common_reference_sizing_and_cash_arithmetic():
    r = run(context(changes={1: {'open': 89., 'high': 89., 'low': 88., 'close': 89.}}), cost='main')
    t = r.trades.iloc[0]
    q = np.floor((500 / 10.17) / .001) * .001
    entry, exit = 100 * 1.0005, 89 * .9998
    fees = q * (entry + exit) * .0005
    assert t.q == pytest.approx(q)
    assert t.entry_price == pytest.approx(entry)
    assert t.exit_price == pytest.approx(exit)
    assert t.fees == pytest.approx(fees)
    assert t.net_pnl == pytest.approx(q * (exit - entry) - fees)
    assert r.equity.iloc[-1] == pytest.approx(100000 + t.net_pnl)
    assert t.net_R == pytest.approx(t.net_pnl / (q * 10))
    assert t.risk_budget_exceeded
    assert t.entry_ref == 100. and t.exit_ref == 89.
    assert t.gross_bps == pytest.approx(-1100.)
    assert t.gross_pnl == pytest.approx(q * (89. - 100.))
    assert t.net_pnl == pytest.approx(t.gross_pnl - t.slippage_cost - t.fees + t.funding_cashflow)


def test_limit_uses_same_market_budget_and_grid_quantity():
    c = context(changes={0: {'low': 97.3}, 1: {'open': 89., 'low': 88., 'high': 89., 'close': 89.}})
    for cost, budget in [('gross', 0), ('main', .0017), ('harsh', .0034)]:
        m = run(c, cost=cost).trades.iloc[0]
        l = run(c, version='L_limit', cost=cost).trades.iloc[0]
        assert l.q == m.q == pytest.approx(np.floor((500 / (10 + 100 * budget)) / .001) * .001)
        assert l.entry_price == 97.5


def test_activation_gap_is_post_only_reject_but_later_gap_fills_then_stops():
    rejected = run(context(changes={0: {'open': 89., 'high': 89., 'low': 88., 'close': 89.}}), version='L_limit')
    assert rejected.trades.empty
    assert rejected.orders.iloc[0].status == 'post_only_rejected'
    later = run(context(changes={1: {'open': 89., 'high': 89., 'low': 88., 'close': 89.}}), version='L_limit')
    t = later.trades.iloc[0]
    assert t.entry_price == 97.5
    assert t.exit_price == 89.
    assert t.net_pnl == pytest.approx((89 - 97.5) * 50)
    assert t.exit_reason == 'gap_stop'
    assert t.entry_time == t.exit_time == START + pd.Timedelta(minutes=1)
    assert t.entry_minute_stop


def test_limit_requires_one_tick_penetration_and_accepts_last_minute():
    miss = run(context(changes={0: {'low': 97.5}}), version='L_limit')
    assert miss.trades.empty
    assert miss.orders.iloc[0].status == 'expired'
    assert miss.orders.iloc[0].wait_minutes == 60
    last = run(context(changes={59: {'low': 97.4}}), version='L_limit')
    assert last.trades.iloc[0].entry_time == START + pd.Timedelta(minutes=59)
    too_late = run(context(changes={60: {'low': 97.0}}), version='L_limit')
    assert too_late.trades.empty


def test_entry_minute_stop_precedes_mean_close_confirmation():
    r = run(context(changes={0: {'low': 89., 'close': 111., 'high': 112.}}), version='L_limit')
    t = r.trades.iloc[0]
    assert t.exit_reason == 'stop'
    assert t.entry_price == 97.5 and t.exit_price == 90.
    assert t.minutes_held == 1


def test_mean_requires_close_and_executes_next_open_with_stop_priority():
    c = context(changes={0: {'high': 115.}, 1: {'close': 111., 'high': 112.},
                         2: {'open': 108., 'high': 109., 'low': 107., 'close': 108.}})
    t = run(c).trades.iloc[0]
    assert t.exit_time == START + pd.Timedelta(minutes=2)
    assert t.exit_reason == 'mean_reversion'
    assert t.exit_price == 108.
    c2 = context(changes={0: {'close': 111., 'high': 112.},
                          1: {'open': 89., 'high': 89., 'low': 88., 'close': 89.}})
    assert run(c2).trades.iloc[0].exit_reason == 'gap_stop'


def test_deadline_is_frozen_at_decision_and_uses_previous_minute_close():
    c = context(changes={59: {'low': 97.4}, 1439: {'close': 111., 'high': 112.},
                         1440: {'open': 150., 'high': 151., 'low': 149., 'close': 150.}})
    t = run(c, version='L_limit').trades.iloc[0]
    assert t.exit_time == START + pd.Timedelta(minutes=1440)
    assert t.exit_minute == START + pd.Timedelta(minutes=1439)
    assert t.exit_reason == 'deadline'
    assert t.exit_price == 111.
    assert t.minutes_held == 1381


def test_expiry_and_post_only_rejection_both_start_four_hour_cooldown():
    r = run(context(n=1900), signals(0, 60, 240, 300), version='L_limit')
    d = r.decisions.set_index('decision_time')
    assert d.loc[START + pd.Timedelta(minutes=60), 'status'] == 'cooldown'
    assert d.loc[START + pd.Timedelta(minutes=240), 'status'] == 'cooldown'
    assert d.loc[START + pd.Timedelta(minutes=300), 'status'] == 'submitted'
    c = context(n=1900, changes={0: {'open': 97., 'low': 96., 'high': 98., 'close': 97.}})
    r2 = run(c, signals(0, 180, 240), version='L_limit')
    d2 = r2.decisions.set_index('decision_time')
    assert d2.loc[START + pd.Timedelta(minutes=180), 'status'] == 'cooldown'
    assert d2.loc[START + pd.Timedelta(minutes=240), 'status'] == 'submitted'


def test_open_exit_and_intraminute_exit_cooldown_use_distinct_boundaries():
    # The same +4h natural-hour parent is accepted only for an opening exit.
    gap = context(n=1900, changes={0: {'open': 89., 'high': 89., 'low': 88., 'close': 89.}})
    dg = run(gap, signals(0, 240)).decisions
    assert dg.iloc[1].status == 'submitted'
    intra = context(n=1900, changes={0: {'low': 89.}})
    di = run(intra, signals(0, 240, 300)).decisions
    assert list(di.status) == ['submitted', 'cooldown', 'submitted']


def test_stress_delays_activation_rounds_penetration_and_charges_taker():
    # L=9750, 2bp=1.95 rounds up to $2.0; 9748.1 cannot fill.
    sig = signals(eref=10000., atr=500., target=11000.)
    c = context(price=10000., changes={0: {'low': 9740.}, 1: {'low': 9748.1},
                                       2: {'low': 9748.}, 3: {'low': 8990.}})
    t = run(c, sig, version='L_limit', cost='main', stress=True).trades.iloc[0]
    assert t.entry_time == START + pd.Timedelta(minutes=2)
    assert t.entry_price == 9750.
    assert t.fees == pytest.approx(t.q * (9750. * .0005 + 9000. * .9998 * .0005))
    with pytest.raises(ValueError):
        run(c, sig, version='M_market', cost='main', stress=True)
    with pytest.raises(ValueError):
        run(c, sig, version='L_limit', cost='gross', stress=True)


def funding_at(minute, rate=.01):
    return pd.Series([rate], index=pd.DatetimeIndex([START + pd.Timedelta(minutes=minute)]), name='funding_rate')


def test_funding_excludes_new_entry_and_includes_existing_open_exit():
    c = context(funding=funding_at(0), changes={1: {'low': 89.}})
    assert run(c, cost='main').trades.iloc[0].funding_cashflow == 0
    c = context(funding=funding_at(1), changes={0: {'close': 111., 'high': 112.},
                                              1: {'open': 108., 'low': 107., 'high': 109., 'close': 108.}})
    t = run(c, cost='main').trades.iloc[0]
    assert t.funding_cashflow == pytest.approx(-t.q * 108 * .01)
    c = context(funding=funding_at(1), changes={0: {'low': 89.}})
    assert run(c, cost='main').trades.iloc[0].funding_cashflow == 0


def test_deadline_close_funding_uses_ending_close_exactly_once():
    c = context(funding=funding_at(1440), changes={1439: {'close': 105., 'high': 106.}})
    t = run(c, cost='main').trades.iloc[0]
    assert t.funding_cashflow == pytest.approx(-t.q * 105 * .01)
    assert t.exit_price == pytest.approx(105 * .9998)


def test_maintenance_cancels_pending_and_forces_next_valid_open():
    c = context(bad=(1, 2), changes={3: {'open': 95., 'high': 96., 'low': 94., 'close': 95.}})
    l = run(c, version='L_limit')
    assert l.trades.empty and l.orders.iloc[0].status == 'maintenance_cancelled'
    m = run(c)
    t = m.trades.iloc[0]
    assert t.exit_reason == 'maintenance'
    assert t.exit_price == 95.
    assert t.halt_in_holding


def test_maintenance_funding_uses_previous_valid_close_and_overrun_is_reported():
    c = context(n=1445, bad=(1439, 1440, 1441), funding=funding_at(1440),
                changes={1438: {'close': 104., 'high': 105.},
                         1442: {'open': 103., 'high': 104., 'low': 102., 'close': 103.}})
    t = run(c, cost='main').trades.iloc[0]
    assert t.funding_cashflow == pytest.approx(-t.q * 104 * .01)
    assert t.overrun_24h and t.exit_reason == 'maintenance'


def test_unresolved_position_and_order_raise_instead_of_eof_liquidation():
    with pytest.raises(RuntimeError, match='unresolved'):
        run(context(n=10))
    with pytest.raises(RuntimeError, match='unresolved'):
        run(context(n=10), version='L_limit')


def test_events_keep_all_eligible_parents_and_unfilled_zero_at_constant_capital():
    from dip_limit_study.engine import simulate_events
    c = context(n=1562, changes={0: {'low': 89.}, 60: {'low': 89.}})
    s = signals(0, 60, 120)
    r = simulate_events(c, s)
    assert len(r) == 3
    assert r.iloc[0].M_net_bps == pytest.approx(r.iloc[1].M_net_bps)
    assert r.iloc[2].L_net_bps == 0
    assert not r.iloc[2].L_filled and r.iloc[2].L_status == 'expired'
    assert np.allclose(r.difference_bps, r.L_net_bps - r.M_net_bps)


def test_calendar_preserves_idle_days_and_does_not_impute_undefined_returns():
    r = run(context(n=3 * 1440))
    assert len(r.daily_equity) == 4
    assert r.daily_returns.iloc[0] == pytest.approx(r.daily_equity.iloc[0] / 100000 - 1)
    assert r.daily_returns.iloc[-1] == 0
    assert len(r.equity) == 3 * 1440


def test_nonpositive_equity_is_permanent_failure_even_if_price_recovers():
    c = context(n=1900, funding=funding_at(1, 25),
                changes={2: {'close': 111., 'high': 112.}})
    r = run(c, signals(0, 300), cost='main')
    assert r.metadata['failed']
    assert 'nonpositive' in r.metadata['failure_reason']
    assert r.decisions.iloc[-1].status == 'failed'


def test_signal_price_validity_and_eligibility_do_not_inspect_activation_open():
    s = signals()
    s.loc[:, 'sl'] = 100.
    with pytest.raises(ValueError, match='frozen'):
        run(context(), s)
    s = signals()
    s.loc[:, 'eligible'] = False
    assert run(context(), s).trades.empty


def test_real_context_gate_is_independent_and_never_borrowed():
    from dataclasses import replace
    from dip_limit_study.scope import RealDataLockedError
    c = context()
    with pytest.raises(RealDataLockedError):
        run(replace(c, source='real'))


def test_minimum_distance_caps_size_without_moving_stop():
    s = signals(eref=1000., atr=.1, target=1010.)
    s.loc[:, 'limit'] = 999.9
    c = context(price=1000., changes={0: {'low': 999.7}})
    t = run(c, s, cost='main').trades.iloc[0]
    assert t.q == 156.25  # 500 / (1.5 + 1.7), below 2x cap.
    assert t.sl == 999.8
    assert t.exit_price == pytest.approx(999.8 * .9998)
    assert run(c, s).trades.iloc[0].q == 200.


def test_pending_gap_crossed_stop_still_uses_open_with_deep_stress_penetration():
    s = signals(eref=10000., atr=.2, target=10010.)
    s.loc[:, 'sl'] = 9999.6
    s.loc[:, 'limit'] = 9999.9
    c = context(price=10000., changes={1: {'open': 10000., 'low': 9999.85},
                                       2: {'open': 9999.5, 'low': 9997.8}})
    t = run(c, s, version='L_limit', cost='main', stress=True).trades.iloc[0]
    assert t.entry_price == 9999.9
    assert t.exit_price == pytest.approx(9999.5 * .9998)
    assert t.exit_reason == 'gap_stop'


def test_bad_activation_is_rejected_without_placeholder_execution():
    for version in ('M_market', 'L_limit'):
        r = run(context(bad=(0,)), version=version)
        assert r.trades.empty
        assert r.orders.iloc[0].status == 'maintenance_rejected'
        assert r.equity.iloc[-1] == 100000.


def test_cancelled_pending_order_also_observes_cooldown():
    r = run(context(n=1900, bad=(1,)), signals(0, 240, 300), version='L_limit')
    assert list(r.decisions.status) == ['submitted', 'cooldown', 'submitted']


@pytest.mark.parametrize('mutation', ['nonhour', 'signal_time', 'index', 'stop', 'limit', 'outside'])
def test_rejects_malformed_or_out_of_context_parent_snapshots(mutation):
    s = signals()
    if mutation == 'nonhour':
        s = signals(1)
    elif mutation == 'signal_time':
        s.loc[:, 'signal_time'] = s.signal_time - pd.Timedelta(hours=1)
    elif mutation == 'index':
        s.index = s.index + pd.Timedelta(hours=1)
    elif mutation == 'stop':
        s.loc[:, 'sl'] = 89.9
    elif mutation == 'limit':
        s.loc[:, 'limit'] = 97.4
    elif mutation == 'outside':
        s = signals(1800)
    with pytest.raises(ValueError):
        run(context(), s)
    from dip_limit_study.engine import simulate_events
    with pytest.raises(ValueError):
        simulate_events(context(), s)


def test_decimal_grid_snapshot_is_accepted_without_binary_one_tick_drift():
    s = signals(eref=103.3, atr=2., target=110.)
    assert s.iloc[0].sl == 99.3 and s.iloc[0]['limit'] == 102.3
    c = context(price=103.3, changes={0: {'low': 102.2}})
    t = run(c, s, version='L_limit', cost='main').trades.iloc[0]
    assert t.entry_ref == t.entry_price == 102.3
    assert t.exit_ref == 103.3
    assert t.gross_bps == pytest.approx((103.3 - 102.3) / 103.3 * 10000)
    assert t.net_pnl == pytest.approx(t.gross_pnl - t.slippage_cost - t.fees)


def test_builder_ineligible_terminal_boundary_remains_accepted():
    from dip_limit_study.features import build_signals
    c = context(n=1500)
    derived = build_signals(c)
    assert derived.iloc[-1].decision_time > c.index[-1]
    assert not derived.iloc[-1].eligible
    r = run(c, derived)
    assert r.trades.empty


def test_empty_and_filled_ledgers_keep_utc_nanosecond_timestamp_types():
    from dip_limit_study.engine import simulate_events
    c = context()
    empty = signals()
    empty.loc[:, 'eligible'] = False
    for s in (empty, signals()):
        r = run(c, s)
        for table, columns in ((r.trades, ['signal_time', 'decision_time', 'entry_time', 'exit_time', 'exit_minute', 'exit_boundary', 'deadline']),
                               (r.orders, ['signal_time', 'decision_time', 'activation_time', 'expiry_time', 'deadline', 'terminal_time', 'terminal_minute', 'fill_time']),
                               (r.decisions, ['signal_time', 'decision_time']),
                               (simulate_events(c, s), ['signal_time', 'decision_time'])):
            for column in columns:
                assert table[column].dtype == pd.DatetimeTZDtype(unit='ns', tz='UTC')
                assert len(table[column].dt.month) == len(table)


def test_stress_pending_latency_is_cancelled_by_bad_decision_minute():
    r = run(context(n=1900, bad=(0,)), signals(0, 60, 240),
            version='L_limit', cost='main', stress=True)
    first = r.orders.iloc[0]
    assert first.status == 'maintenance_cancelled'
    assert first.terminal_time == START
    assert first.wait_minutes == 0
    assert list(r.decisions.status) == ['submitted', 'cooldown', 'submitted']
    assert r.trades.empty and r.equity.iloc[-1] == 100000.


def test_stress_exact_seven_tick_penetration_does_not_round_up_to_eight():
    s = signals(eref=4000., atr=1000., target=5000.)
    c = context(price=4000., changes={1: {'low': 3499.3}})
    r = run(c, s, version='L_limit', cost='main', stress=True)
    assert r.orders.iloc[0].penetration == .7
    assert len(r.trades) == 1
    assert r.trades.iloc[0].entry_time == START + pd.Timedelta(minutes=1)
    assert r.trades.iloc[0].entry_price == 3500.
    from dip_limit_study.engine import _stress_penetration
    assert _stress_penetration(3500.1) == .8


def test_quantity_exact_thousand_and_first_step_is_not_lost():
    from dip_limit_study.config import COSTS
    from dip_limit_study.engine import _quantity
    row = {'eref': 100., 'sl': 90.}
    # .005 * 2002 / 10 = 1.001 exactly in the raw sizing model.
    assert _quantity(2002., row, COSTS['gross']) == 1.001
    # No epsilon may promote a genuinely lower raw quantity to that step.
    assert _quantity(2001.999999999, row, COSTS['gross']) == 1.000

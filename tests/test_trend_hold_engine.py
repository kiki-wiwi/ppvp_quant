from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _fixture(*, trend=(True,), budget=None, opens=None, closes=None, bad=(),
             gap=(), rates=None):
    from trend_hold_study.config import MAIN_START
    from trend_hold_study.scope import Context

    index = pd.date_range(MAIN_START - pd.Timedelta(minutes=1),
                          periods=len(trend) * 1440 + 1, freq="min")
    open_ = np.full(len(index), 100.0)
    close = open_.copy()
    for minute, price in (opens or {}).items():
        open_[minute + 1] = price
    for minute, price in (closes or {}).items():
        close[minute + 1] = price
    is_bad = np.zeros(len(index), dtype=bool)
    is_gap = is_bad.copy()
    is_bad[np.asarray(bad, dtype=int) + 1] = True
    is_gap[np.asarray(gap, dtype=int) + 1] = True
    rate = np.full(len(index), np.nan)
    for minute, value in (rates or {}).items():
        rate[minute + 1] = value
    ctx = Context(index, open_, np.maximum(open_, close), np.minimum(open_, close),
                  close, np.ones(len(index)), is_bad, is_gap, rate)
    days = index[1::1440]
    previous = np.arange(len(days)) * 1440
    features = pd.DataFrame({
        "eref": close[previous],
        "sma60": 100.0,
        "sigma": 0.4,
        "budget": 0.5 if budget is None else budget,
        "g": trend,
        "budget_update": (np.arange(len(days)) == 0) | (days.dayofweek == 0),
        "n_valid_prior": 1440,
        "has_archive_gap": False,
    }, index=days)
    return ctx, features


def _run(fixture, *, version="T_daily_trend", cost="gross_0"):
    from trend_hold_study.config import COSTS
    from trend_hold_study.engine import run_portfolio

    return run_portfolio(*fixture, version, COSTS[cost])


def test_funding_charges_old_quantity_after_target_freeze_and_before_net_fill():
    result = _run(_fixture(trend=(True, True), budget=(0.5, 0.25),
                           opens={0: 110.0, 1440: 120.0},
                           rates={0: 0.5, 1440: 0.01}), cost="main_17")

    assert len(result.orders) == 3
    first, monday, terminal = (row for _, row in result.orders.iterrows())
    assert first.qty_target == 500.0
    assert first.fill_price == pytest.approx(110.055)
    assert first.fee == pytest.approx(27.51375)
    assert result.daily_equity.iloc[0] == pytest.approx(94944.98625)
    assert monday.qty_before == 500.0
    assert monday.qty_target == pytest.approx(237.362465625)
    assert monday.delta_qty == pytest.approx(-262.637534375)
    assert monday.fill_price == pytest.approx(119.976)
    assert result.funding.qty_before.tolist() == [0.0, 500.0]
    assert result.funding.cash_flow.tolist() == [0.0, -600.0]
    assert terminal.terminal
    assert terminal.fill_price == pytest.approx(99.98)
    monday_pnl = 262.637534375 * 19.976
    expected_final = 94944.98625 - 600 + monday_pnl - monday.fee
    expected_final -= 237.362465625 * 0.02 + terminal.fee
    assert result.minute.equity.iloc[-1] == pytest.approx(expected_final)
    assert result.daily_returns.iloc[0] == pytest.approx(-0.0505501375)
    assert result.metadata["total_fees"] == pytest.approx(result.orders.fee.sum())
    assert result.metadata["total_funding"] == -600.0
    assert result.metadata["price_pnl"] == pytest.approx(252.7506875)
    assert result.metadata["turnover_notional"] == pytest.approx(
        (result.orders.delta_qty.abs() * result.orders.fill_price).sum())
    assert result.metadata["total_turnover_fraction"] == pytest.approx(
        result.orders.turnover_fraction.sum())


def test_sunday_entry_and_monday_trend_flip_coincide_with_one_net_exit():
    fixture = _fixture(trend=(True, False, False), budget=(0.5, 0.25, 0.25))
    trend = _run(fixture)
    persistent = _run(fixture, version="P_continuous_long")

    assert trend.orders.delta_qty.tolist() == [500.0, -500.0]
    assert trend.orders.time.tolist() == list(fixture[1].index[:2])
    assert trend.decisions.g.tolist() == [True, False, False]
    assert trend.decisions.budget_update.tolist() == [True, True, False]
    assert trend.decisions.trend_flip.tolist() == [False, True, False]
    assert trend.decisions.pending_after.tolist() == [False, False, False]
    assert persistent.orders.delta_qty.tolist() == [500.0, -250.0, -250.0]
    assert persistent.decisions.g.tolist() == [True, True, True]
    assert fixture[1].g.tolist() == [True, False, False]


def test_bad_midnight_keeps_position_and_refreezes_after_fallback_funding():
    result = _run(_fixture(trend=(True, True, True), budget=(0.5, 0.25, 0.25),
                           bad=(1440,), gap=(1440,), closes={1440: 999.0, 2879: 125.0},
                           opens={1440: 900.0}, rates={1440: 0.01}), cost="main_17")

    assert result.decisions.status.tolist() == ["executed", "canceled_bad", "executed"]
    assert result.decisions.pending_after.tolist() == [False, True, False]
    assert result.decisions.retry.tolist() == [False, False, True]
    assert result.minute.qty.iloc[1440:2880].eq(500.0).all()
    assert result.minute.mark_price.iloc[1440] == 100.0
    assert result.funding.reference_price.tolist() == [100.0]
    assert result.funding.cash_flow.tolist() == [-500.0]
    assert result.metadata["maintenance_held_minutes"] == 1
    assert result.metadata["gap_held_minutes"] == 1
    tuesday = result.orders.iloc[1]
    assert tuesday.qty_target == pytest.approx(223.899975)
    assert tuesday.qty_target != result.decisions.qty_target.iloc[1]


def test_retry_uses_current_trend_and_zero_delta_clears_pending():
    result = _run(_fixture(trend=(True, False, True, True), bad=(1440,)))

    assert result.decisions.status.tolist() == ["executed", "canceled_bad", "zero_delta", "hold"]
    assert result.decisions.pending_after.tolist() == [False, True, False, False]
    assert result.orders.delta_qty.tolist() == [500.0, -500.0]
    assert result.orders.terminal.tolist() == [False, True]


def test_flat_weekly_budget_does_not_create_order_and_position_survives_drawdown():
    result = _run(_fixture(trend=(True,) * 5, closes={4000: 1.0}, rates={1920: 0.05}))

    assert result.minute.equity.iloc[4000] == 50500.0
    assert result.minute.qty.iloc[4 * 1440] == 500.0
    assert result.minute.qty.iloc[-1] == 0.0
    assert result.daily_equity.iloc[-1] == 100000.0
    assert result.decisions.status.tolist() == ["executed", "zero_delta", "hold", "hold", "hold"]
    assert result.orders.terminal.tolist() == [False, True]
    assert result.funding.cash_flow.tolist() == [0.0]


def test_actual_leverage_above_cap_is_reported_without_resizing():
    result = _run(_fixture(budget=(1.0,), opens={0: 110.0}))

    assert result.decisions.reference_leverage.iloc[0] == 1.0
    assert result.minute.qty.iloc[0] == 1000.0
    assert result.minute.actual_leverage.iloc[0] == pytest.approx(100000 / 90000)
    assert result.metadata["minutes_above_leverage_cap"] == 1439


def test_bad_terminal_minute_and_insolvency_stop_the_run():
    with pytest.raises(ValueError, match="terminal"):
        _run(_fixture(bad=(1439,)))
    with pytest.raises(ValueError, match="equity"):
        _run(_fixture(budget=(1.0,), opens={0: 200.0}))
    with pytest.raises(ValueError, match="equity"):
        _run(_fixture(budget=(1.0,), rates={480: 1.0}), cost="main_17")


def test_rejects_missing_or_invalid_daily_inputs_and_real_context_before_access():
    from trend_hold_study.config import COSTS, T_VERSION
    from trend_hold_study.engine import run_portfolio
    from trend_hold_study.scope import RealDataLockedError

    ctx, features = _fixture(trend=(True, True, True))
    with pytest.raises(ValueError):
        _run((ctx, features.iloc[1:]))
    for column, value in (("budget", 1.1), ("sigma", np.nan), ("eref", 101.0),
                          ("budget_update", False)):
        invalid = features.copy()
        invalid.loc[features.index[0], column] = value
        with pytest.raises(ValueError):
            _run((ctx, invalid))
    with pytest.raises(RealDataLockedError):
        run_portfolio(SimpleNamespace(source="real"), None, T_VERSION, COSTS["gross_0"])


def test_terminal_closes_at_the_final_close_within_the_sample():
    ctx, features = _fixture(opens={1439: 85.0}, closes={1439: 130.0},
                             rates={960: -0.01})
    result = _run((ctx, features))
    terminal = result.orders.iloc[-1]

    assert terminal.reference_price == 130.0
    assert terminal.time == ctx.index[-1] + pd.Timedelta(minutes=1) - pd.Timedelta(nanoseconds=1)
    assert result.funding.time.tolist() == [ctx.index[961]]
    assert result.minute.qty.iloc[-1] == 0.0
    assert result.minute.held_qty.iloc[-1] == 500.0
    assert result.metadata["held_minutes"] == 1440
    assert result.daily_equity.iloc[-1] == 115000.0


def test_account_failure_preserves_processed_ledgers_and_failure_state():
    ctx, features = _fixture(budget=(1.0,), rates={480: 1.0})
    with pytest.raises(ValueError, match="equity") as stopped:
        _run((ctx, features), cost="main_17")

    partial = stopped.value.partial
    assert partial["failure_time"] == ctx.index[481]
    assert partial["last_confirmed_time"] == ctx.index[480]
    assert partial["last_confirmed_equity"] == pytest.approx(99899.975)
    assert partial["orders"].delta_qty.tolist() == [1000.0]
    assert partial["decisions"].status.tolist() == ["executed"]
    assert partial["funding"].cash_flow.tolist() == [-100000.0]

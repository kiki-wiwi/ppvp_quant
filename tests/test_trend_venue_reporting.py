"""Pure synthetic metric and continuous-boundary slicing checks."""
import math

import numpy as np
import pandas as pd
import pytest

from trend_hold_venue_study.models import Result
from trend_hold_venue_study.reporting import metrics, period_slice


def fixture_result(equity, returns, start='2025-06-30', initial=100_000., coin=None):
    days = pd.date_range(start, periods=len(equity), freq='D', tz='UTC').as_unit('ns')
    minute_index = days + pd.Timedelta(hours=23, minutes=59)
    minute = pd.DataFrame({'equity': equity, 'bad': False}, index=minute_index)
    metadata = {'version': 'V3_CM_DeltaMatched', 'cost': 'main_17', 'asset': 'ETH',
                'initial_capital': initial, 'source': 'synthetic', 'liquidated': False}
    if coin is not None:
        minute['coin_equity'] = coin
        metadata.update(initial_coin=1.5, final_coin=coin[-1], coin_growth=coin[-1] / 1.5 - 1.)
    return Result(minute, pd.Series(equity, index=days, name='equity'),
                  pd.Series(returns, index=days, name='return'),
                  pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), metadata)


def test_metrics_use_registered_population_downside_and_sample_volatility():
    result = fixture_result([110., 99.], [.1, -.1], initial=100., coin=[2., 3.])
    bad_and_early = pd.DataFrame(
        {'equity': [95., 1_000.], 'bad': [False, True], 'coin_equity': [1.5, 1.5]},
        index=pd.DatetimeIndex(['2025-06-30 00:00Z', '2025-06-30 00:01Z']).as_unit('ns'))
    result.minute = pd.concat([bad_and_early, result.minute]).sort_index()
    summary = metrics(result, start_equity=100.)
    assert summary['total_return'] == pytest.approx(-.01, rel=0., abs=1e-15)
    assert summary['realized_volatility'] == pytest.approx(math.sqrt(7.3), rel=0., abs=1e-14)
    assert summary['sharpe'] == 0.
    assert summary['sortino'] == 0.
    assert summary['max_drawdown'] == pytest.approx(.1, rel=0., abs=1e-15)
    expected_cagr = .99 ** (365. / 2.) - 1.
    assert summary['cagr'] == pytest.approx(expected_cagr, rel=0., abs=1e-14)
    assert summary['calmar'] == pytest.approx(expected_cagr / .1, rel=0., abs=1e-12)
    assert summary['coin_growth'] == 1.
    assert summary['asset'] == 'ETH'
    assert summary['n_days'] == 2


def test_drawdown_includes_initial_capital_before_first_observed_close():
    result = fixture_result([90., 99.], [-.1, .1], initial=100.)
    assert metrics(result, start_equity=100.)['max_drawdown'] == pytest.approx(.1, rel=0., abs=1e-15)


def test_sortino_divides_by_all_days_including_nonnegative_days():
    result = fixture_result([90., 90., 108.], [-.1, 0., .2], initial=100.)
    expected = math.sqrt(365.) * (.1 / 3.) / math.sqrt(.01 / 3.)
    assert metrics(result, start_equity=100.)['sortino'] == pytest.approx(expected, rel=0., abs=1e-13)


def test_zero_equity_marks_bankruptcy_once_then_zero_returns():
    result = fixture_result([0., 0.], [-1., 0.], initial=100.)
    result.metadata['liquidated'] = True
    summary = metrics(result, start_equity=100.)
    assert summary['total_return'] == -1.
    assert summary['max_drawdown'] == 1.
    assert math.isnan(summary['cagr'])
    assert math.isnan(summary['calmar'])
    assert summary['sharpe'] == pytest.approx(-math.sqrt(365. / 2.), rel=0., abs=1e-13)
    assert summary['sortino'] == pytest.approx(-math.sqrt(365. / 2.), rel=0., abs=1e-13)
    assert summary['liquidated'] is True
    after = period_slice(result, pd.Timestamp('2025-07-01', tz='UTC'),
                         pd.Timestamp('2025-07-02', tz='UTC'))
    assert after['start_equity'] == 0.
    assert after['daily_returns'].iloc[0] == 0.
    assert math.isnan(after['total_return'])  # No invented denominator after ruin.


def test_flat_returns_and_zero_drawdown_have_registered_na_ratios():
    result = fixture_result([100., 100.], [0., 0.], initial=100.)
    summary = metrics(result, start_equity=100.)
    assert summary['total_return'] == summary['cagr'] == summary['realized_volatility'] == 0.
    assert summary['max_drawdown'] == 0.
    assert math.isnan(summary['sharpe'])
    assert math.isnan(summary['sortino'])
    assert math.isnan(summary['calmar'])


def test_continuous_eth_july_split_retains_prior_day_denominator_and_original_returns():
    result = fixture_result([120_000., 132_000., 118_800.], [.2, .1, -.1], coin=[2., 2.2, 1.98])
    original_equity = result.daily_equity.copy(deep=True)
    original_returns = result.daily_returns.copy(deep=True)
    val = period_slice(result, pd.Timestamp('2025-06-30', tz='UTC'),
                       pd.Timestamp('2025-07-01', tz='UTC'))
    oos = period_slice(result, pd.Timestamp('2025-07-01', tz='UTC'),
                       pd.Timestamp('2025-07-03', tz='UTC'))
    assert val['start_equity'] == 100_000.
    assert val['final_equity'] == 120_000.  # No hypothetical June-end exit or FX fee.
    assert val['daily_returns'].iloc[0] == .2
    assert oos['start_equity'] == 120_000.
    assert oos['prior_day'] == pd.Timestamp('2025-06-30', tz='UTC')
    assert oos['daily_returns'].iloc[0] == .1
    assert oos['final_equity'] == 118_800.
    assert oos['total_return'] == pytest.approx(-.01, rel=0., abs=1e-15)
    assert oos['start_coin'] == 2.
    assert oos['final_coin'] == 1.98
    assert oos['coin_growth'] == pytest.approx(-.01, rel=0., abs=1e-15)
    pd.testing.assert_series_equal(oos['daily_returns'], original_returns.iloc[1:])
    pd.testing.assert_series_equal(oos['daily_coin'],
                                   pd.Series([2.2, 1.98], index=original_equity.index[1:], name='coin_equity'))
    pd.testing.assert_series_equal(result.daily_equity, original_equity)
    pd.testing.assert_series_equal(result.daily_returns, original_returns)
    oos['daily_equity'].iloc[0] = 1.
    assert result.daily_equity.iloc[1] == 132_000.


def test_period_bounds_cannot_implicitly_reset_or_skip_missing_calendar_days():
    result = fixture_result([120_000., 132_000., 118_800.], [.2, .1, -.1])
    with pytest.raises(ValueError):
        period_slice(result, '2025-07-01', '2025-07-03')
    with pytest.raises(ValueError):
        period_slice(result, pd.Timestamp('2025-07-01 01:00', tz='UTC'),
                     pd.Timestamp('2025-07-03', tz='UTC'))
    with pytest.raises(ValueError):
        period_slice(result, pd.Timestamp('2025-06-29', tz='UTC'),
                     pd.Timestamp('2025-07-03', tz='UTC'))
    result.daily_equity = result.daily_equity.iloc[[0, 2]]
    result.daily_returns = result.daily_returns.iloc[[0, 2]]
    with pytest.raises(ValueError):
        metrics(result)


def test_nonfinite_inputs_and_invalid_zero_recovery_stop_reporting():
    result = fixture_result([0., 100.], [-1., 0.], initial=100.)
    with pytest.raises(ValueError):
        metrics(result, start_equity=100.)
    result = fixture_result([100., 101.], [0., np.nan], initial=100.)
    with pytest.raises(ValueError):
        metrics(result, start_equity=100.)


def test_nonzero_constant_returns_have_exact_zero_volatility_and_na_sharpe():
    returns = np.full(81, .1, dtype=float)
    equity = 100. * np.cumprod(1. + returns)
    result = fixture_result(equity, returns, initial=100.)
    summary = metrics(result, start_equity=100.)
    assert summary['realized_volatility'] == 0.
    assert math.isnan(summary['sharpe'])
    assert math.isnan(summary['sortino'])
    assert summary['total_return'] > 0.

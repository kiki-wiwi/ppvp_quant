"""Synthetic-only CP3 slices and descriptive ledger aggregation."""
import math

import numpy as np
import pandas as pd
import pytest

from trend_hold_venue_study import cp3_analysis as analysis
from trend_hold_venue_study.models import Result


def example_result(zero=False):
    days = pd.date_range('2025-06-29', periods=4, tz='UTC').as_unit('ns')
    values = [110000., 120000., 126000., 132000.]
    coins = [1.1, 1.2, 1.26, 1.32]
    if zero:
        values, coins = [110000., 0., 0., 0.], [1.1, 0., 0., 0.]
    previous = np.r_[100000., values[:-1]]
    returns = np.divide(values, previous, out=np.ones(4), where=previous > 0) - 1.
    minute = pd.DataFrame({'equity': values, 'bad': False, 'coin_equity': coins,
        'actual_leverage': [1., 1., .5, 0.]},
        index=days + pd.Timedelta(hours=23, minutes=59))
    return Result(minute, pd.Series(values, index=days), pd.Series(returns, index=days),
        pd.DataFrame({'time': [days[0], days[2]], 'fee': [1., 2.], 'slippage_cost': [.5, 1.]}),
        pd.DataFrame(), pd.DataFrame({'time': [days[2]], 'cash_flow': [3.]}),
        {'initial_capital': 100000., 'initial_coin': 1., 'final_coin': coins[-1],
         'coin_growth': coins[-1] - 1., 'final_equity': values[-1], 'version': 'V1_USDM',
         'cost': 'main_17', 'asset': 'BTC', 'window': 'Post-IS', 'order_count': 2})


def test_slice_preserves_actual_preceding_equity_coin_and_existing_rows():
    result = example_result()
    sliced = analysis.account_slice(result, '2025-07-01T00:00Z', '2025-07-03T00:00Z')
    assert sliced['metrics']['start_equity'] == 120000.
    assert sliced['metrics']['start_coin'] == 1.2
    assert sliced['metrics']['total_return'] == 132000. / 120000. - 1.
    assert sliced['metrics']['coin_growth'] == 1.32 / 1.2 - 1.
    pd.testing.assert_series_equal(sliced['result'].daily_returns, result.daily_returns.iloc[2:])
    pd.testing.assert_frame_equal(sliced['result'].minute, result.minute.iloc[2:])
    assert len(sliced['result'].orders) == 1
    assert sliced['result'].orders['fee'].iloc[0] == 2.
    assert result.metadata['initial_coin'] == 1.


def test_zero_preceding_wealth_stays_undefined_without_restarting_account():
    sliced = analysis.account_slice(example_result(True), '2025-07-01T00:00Z', '2025-07-03T00:00Z')
    for name in ('total_return', 'cagr', 'coin_growth', 'sharpe', 'sortino', 'max_drawdown', 'calmar'):
        assert math.isnan(sliced['metrics'][name]), name
    assert sliced['metrics']['start_equity'] == 0.
    assert sliced['metrics']['final_equity'] == 0.
    assert (sliced['result'].daily_returns == 0.).all()


def test_ledger_daily_reconciles_observed_linear_cashflows():
    result = example_result()
    daily = analysis.daily_accounting(result)
    assert daily.loc['2025-07-01', 'funding_usd'] == 3.
    assert daily.loc['2025-07-01', 'fee_usd'] == 2.
    assert daily.loc['2025-07-01', 'slippage_usd'] == -1.
    assert daily.loc['2025-07-01', 'price_pnl_usd'] == 6000.
    assert daily['equity_change_usd'].sum() == 32000.
    assert np.max(np.abs(daily['reconciliation_error_usd'])) < 1e-9


def test_chart_daily_uses_only_effective_minutes_and_exact_trough_time():
    result = example_result()
    first = result.minute.index[0]
    extra = result.minute.iloc[[0]].copy()
    extra.index = pd.DatetimeIndex([first - pd.Timedelta(minutes=1)])
    extra['equity'], extra['bad'] = 1., True
    result.minute = pd.concat([extra, result.minute])
    chart = analysis.chart_daily(result)
    assert chart['equity_drawdown_minute_min'].min() == 0.
    assert chart.iloc[0]['equity_drawdown_minute_min_time'] == first
    assert chart.iloc[0]['coin_equity_at_minute_dd_min'] == 1.1


def test_compound_reference_matches_asset_version_cost_and_keeps_independent_label():
    common = {'asset': 'BTC', 'version': 'V1_USDM', 'cost': 'main_17',
              'signal_source': 'UM', 'whole_contracts': False}
    old = pd.DataFrame([{**common, 'window': 'IS-Common', 'total_return': .2}])
    post = pd.DataFrame([{**common, 'window': 'Post-IS', 'total_return': .3}])
    result = analysis.combined_reference(old, post)
    assert result.iloc[0]['combined_reference_return'] == 1.2 * 1.3 - 1.
    assert not bool(result.iloc[0]['continuous_positions'])
    with pytest.raises(ValueError, match='duplicate'):
        analysis.combined_reference(pd.concat([old, old]), post)


def test_chop_uses_observed_negative_episode_price_loss_plus_actual_friction():
    days = pd.date_range('2024-02-29', '2024-11-01', tz='UTC').as_unit('ns')
    equity = pd.Series(100000., index=days)
    minute = pd.DataFrame({'equity': equity.to_numpy(), 'bad': False}, index=days + pd.Timedelta(hours=23, minutes=59))
    empty = pd.DataFrame()
    result = Result(minute, equity, pd.Series(0., index=days), empty, empty, empty,
        {'version': 'V1_USDM', 'cost': 'main_17', 'initial_capital': 100000., 'final_equity': 100000.})
    features = pd.DataFrame({'g': days < pd.Timestamp('2024-07-01', tz='UTC')}, index=days)
    accounting = pd.DataFrame(0., index=days, columns=['price_pnl_usd', 'contract_price_pnl_usd',
        'funding_usd', 'fee_usd', 'slippage_usd', 'conversion_usd', 'friction_usd'])
    accounting.loc['2024-03-01', ['price_pnl_usd', 'contract_price_pnl_usd']] = -100.
    accounting.loc['2024-03-02', ['price_pnl_usd', 'contract_price_pnl_usd']] = 40.
    accounting.loc['2024-07-01', ['price_pnl_usd', 'contract_price_pnl_usd']] = 25.
    accounting.loc['2024-03-01', ['funding_usd', 'fee_usd', 'friction_usd']] = [12., 20., 20.]
    job = {'asset': 'BTC', 'window': 'Post-IS', 'signal_source': 'UM', 'version': 'V1_USDM',
           'cost': 'main_17', 'whole_contracts': False, 'job_id': 'synthetic'}
    diagnostic = analysis.chop_diagnosis(result, features, job, accounting)
    row = diagnostic['summary']
    assert row['negative_episode_price_loss_usd'] == 60.
    assert row['negative_on_episode_price_loss_usd'] == 60.
    assert row['funding_offset_denominator_usd'] == 80.
    assert row['funding_offset_ratio'] == 12. / 80.
    assert row['start_equity'] == 100000.
    assert diagnostic['episodes'].iloc[0]['start_boundary_clipped']
    assert diagnostic['episodes'].iloc[-1]['end_boundary_clipped']
    assert row['switch_count'] == 1
    assert row['actual_order_count'] == 0
    assert row['episode_basis'] == 'canonical-trend-state episodes'
    passive = analysis.chop_diagnosis(result, features, {**job, 'version': 'P_USDM'}, accounting)
    assert passive['summary']['canonical_switch_count'] == 1
    assert passive['summary']['actual_order_count'] == 0
    assert passive['summary']['episode_basis'] == 'canonical-trend-state episodes'


def test_post_periods_stop_at_may_end_and_keep_actual_july_denominators():
    days = pd.date_range('2024-01-01', '2026-06-03', inclusive='left', tz='UTC').as_unit('ns')
    equity = pd.Series(110000., index=days)
    equity.loc['2025-06-30'] = 120000.
    equity.loc['2025-07-01':] = 126000.
    equity.loc['2026-05-31'] = 132000.
    equity.loc['2026-06-01':] = 999000.
    values = equity.to_numpy()
    returns = pd.Series(values / np.r_[100000., values[:-1]] - 1., index=days)
    minute = pd.DataFrame({'equity': values, 'bad': False, 'coin_equity': values / 100000.},
        index=days + pd.Timedelta(hours=23, minutes=59))
    orders = pd.DataFrame({'time': pd.to_datetime(['2025-06-30T23:59Z', '2025-07-01T00:00Z',
        '2026-05-31T23:59Z', '2026-06-01T00:00Z'], utc=True),
        'fee': [5., 7., 11., 13.], 'slippage_cost': 0.})
    result = Result(minute, equity, returns, orders, pd.DataFrame(), pd.DataFrame(),
        {'initial_capital': 100000., 'initial_coin': 1., 'final_coin': 9.99,
         'final_equity': 999000., 'coin_growth': 8.99, 'version': 'V1_USDM'})
    job = {'asset': 'BTC', 'window': 'Post-IS', 'signal_source': 'UM', 'version': 'V1_USDM',
           'cost': 'main_17', 'whole_contracts': False, 'job_id': 'synthetic'}
    periods = analysis.period_metrics(result, job).set_index('period')
    assert periods['n_days'].to_dict() == {'Post-IS': 882, 'VAL': 547, 'OOS': 335}
    assert periods.loc['Post-IS', 'end'] == periods.loc['OOS', 'end'] == pd.Timestamp('2026-06-01T00:00Z')
    assert periods.loc['VAL', 'end'] == periods.loc['OOS', 'start'] == pd.Timestamp('2025-07-01T00:00Z')
    assert periods.loc['OOS', 'prior_day'] == pd.Timestamp('2025-06-30T00:00Z')
    assert periods.loc['OOS', 'start_equity'] == 120000.
    assert periods.loc['OOS', 'start_coin'] == 1.2
    assert periods.loc['OOS', 'final_equity'] == 132000.
    assert periods.loc['OOS', 'final_coin'] == 1.32
    assert periods.loc['OOS', 'total_return'] == pytest.approx(.1)
    assert periods.loc['OOS', 'coin_growth'] == pytest.approx(.1)
    assert periods.loc['OOS', 'cagr'] == pytest.approx(1.1 ** (365. / 335.) - 1.)
    assert periods['order_count'].to_dict() == {'Post-IS': 3, 'VAL': 1, 'OOS': 2}
    assert periods.loc['OOS', 'fee_usd'] == 18.
    assert result.metadata['initial_capital'] == 100000.
    assert result.metadata['final_equity'] == 999000.
    assert result.minute.index[-1] == pd.Timestamp('2026-06-02T23:59Z')


def test_post_period_metrics_keep_short_synthetic_history_and_split_behavior():
    result = example_result()
    job = {'asset': 'BTC', 'window': 'Post-IS', 'version': 'V1_USDM'}
    periods = analysis.period_metrics(result, job).set_index('period')
    assert periods['n_days'].to_dict() == {'Post-IS': 4, 'VAL': 2, 'OOS': 2}
    assert periods.loc['OOS', 'end'] == pd.Timestamp('2025-07-03T00:00Z')
    assert periods.loc['OOS', 'start_equity'] == 120000.
    assert periods.loc['OOS', 'total_return'] == 132000. / 120000. - 1.

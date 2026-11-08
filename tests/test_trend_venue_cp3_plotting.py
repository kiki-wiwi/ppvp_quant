"""Synthetic CP3 figures: no market readers, engines, inference or real outputs."""
from copy import deepcopy

import matplotlib.dates as mdates
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from trend_hold_venue_study.cp3_plotting import build_figures, render_outputs


VERSIONS = ('V1_USDM', 'P_USDM', 'V2_SPOT', 'P_SPOT',
            'V3_CM_DeltaMatched', 'P_CM_DeltaMatched',
            'V4_CM_CoinAccum', 'HODL_1x_Coin')
EXPECTED_FILES = ['fig_1.png', 'fig_2.png', 'fig_3.png', 'fig_4.png']


def fixture():
    accounts, comparison, periods = {}, [], []
    for asset in ('BTC', 'ETH'):
        for window in ('IS-Common', 'Post-IS'):
            dates = pd.to_datetime(
                ['2020-09-01', '2021-06-01', '2022-01-01', '2022-11-01',
                 '2023-06-01', '2023-12-31'] if window == 'IS-Common' else
                ['2024-01-01', '2024-03-01', '2024-10-31', '2025-06-30',
                 '2025-07-01', '2026-05-31'], utc=True)
            for position, version in enumerate(VERSIONS):
                factor = position + 1.
                job_id = '__'.join((asset, window, 'UM', version, 'main_17', 'continuous'))
                job = {'job_id': job_id, 'asset': asset, 'window': window,
                       'signal_source': 'UM', 'version': version,
                       'cost': 'main_17', 'whole_contracts': False}
                chart = pd.DataFrame({
                    'equity': [100_000., 120_000., 112_000., 135_000., 140_000., 150_000.],
                    'return': [0., .2, -.0666666667, .2053571429, .0370370370, .0714285714],
                    'equity_drawdown_minute_min': [-.01, -.55, -.12, -.08, -.04, -.02],
                    'equity_drawdown_minute_max': [0., -.02, -.01, 0., -.01, 0.],
                    'equity_drawdown_minute_min_time': dates + pd.Timedelta(hours=12, minutes=7),
                    'funding_usd': np.array([1., 2., -1., 4., 3., -2.]) * factor,
                    'funding_cumulative_usd': np.array([1., 3., 2., 6., 9., 7.]) * factor,
                    'actual_leverage_mean': [.2, .3, .1, .4, .5, .1],
                    'actual_leverage_min': [.1, .2, 0., .3, .4, 0.],
                    'actual_leverage_max': [.3, .4, .2, .5, .6, .2],
                }, index=dates)
                metadata = {'initial_capital': 100_000., 'liquidated': False}
                pressure = pd.DataFrame()
                if version in ('V4_CM_CoinAccum', 'HODL_1x_Coin'):
                    metadata['initial_coin'] = 2.
                    chart['coin_equity'] = [2., 2.8, 2.6, 3., 3.2, 3.4]
                    chart['coin_equity_at_minute_dd_min'] = [1.9, 1.1, 2.4, 2.8, 3., 3.3]
                    if version == 'V4_CM_CoinAccum':
                        pressure = pd.DataFrame({'time': [dates[2] + pd.Timedelta(hours=3)],
                                                 'residual_coin': [.5]})
                        metadata['liquidated'] = True
                daily = chart[['equity', 'return']].copy()
                if 'coin_equity' in chart:
                    daily['coin_equity'] = chart['coin_equity']
                accounts[job_id] = {'job': job, 'metadata': metadata, 'daily': daily,
                                    'chart_daily': chart, 'pressure_events': pressure}
                comparison.append({**job, 'sharpe': .6 + .1 * position,
                                   'max_drawdown': .55, 'funding_usd': 7. * factor})
                for period, funding in ([('IS-Common', 7.)] if window == 'IS-Common'
                                        else [('Post-IS', 7.), ('VAL', 6.), ('OOS', 1.)]):
                    periods.append({**job, 'period': period, 'funding_usd': funding * factor,
                                    'sharpe': .6 + .1 * position, 'max_drawdown': .55,
                                    'start_equity': 100_000.})
    diagnostics = {'period_metrics': pd.DataFrame(periods)}
    return accounts, {}, {}, pd.DataFrame(comparison), diagnostics


def _close(figures):
    for figure in figures.values():
        plt.close(figure)


def test_exact_intraday_dd_extreme_and_continuous_post_path_are_visible():
    # A daily-close-derived DD or dropped intraday trough must fail this test.
    args = fixture()
    figures = build_figures(*args)
    try:
        equity, drawdown = figures['fig_1.png'].axes
        assert equity.get_ylabel() == 'Cash-comparable equity (USD thousands)'
        v1_line = next(line for line in equity.lines if line.get_label() == 'V1 UM')
        np.testing.assert_array_equal(v1_line.get_ydata(), [100., 120., 112., 135., 140., 150.])
        assert pd.Timestamp(v1_line.get_xdata()[-1]) == pd.Timestamp('2026-05-31T00:00:00Z')
        v1_extreme = next(mark for mark in drawdown.collections
                          if mark.get_label() == 'V1 UM maximum drawdown')
        point = v1_extreme.get_offsets()[0]
        assert float(point[0]) == pytest.approx(mdates.date2num(pd.Timestamp('2024-03-01T12:07:00Z')))
        assert float(point[1]) == pytest.approx(-55.)
        assert 'effective 1m' in drawdown.get_title(loc='left')
        split = pd.Timestamp('2025-07-01T00:00:00Z')
        split_lines = [line for line in equity.lines if line.get_label() == 'VAL / OOS boundary']
        assert len(split_lines) == 1
        assert pd.Timestamp(split_lines[0].get_xdata()[0]) == split
        assert any(patch.get_label() == '2024-03 to 2024-10 chop window' for patch in equity.patches)
    finally:
        _close(figures)


def test_funding_panels_reset_only_between_independent_accounts_and_use_supplied_slices():
    # Joining IS/Post or rebasing OOS funding would lose the supplied continuous path.
    args = fixture()
    figures = build_figures(*args)
    try:
        axes = figures['fig_2.png'].axes
        is_line = next(line for line in axes[0].lines if line.get_label() == 'V1 UM')
        post_line = next(line for line in axes[1].lines if line.get_label() == 'V1 UM')
        np.testing.assert_allclose(is_line.get_ydata(), np.array([1., 3., 2., 6., 9., 7.]) / 1000.)
        np.testing.assert_allclose(post_line.get_ydata(), np.array([1., 3., 2., 6., 9., 7.]) / 1000.)
        assert 'independent' in axes[0].get_title(loc='left').lower()
        assert 'continuous' in axes[1].get_title(loc='left').lower()
        v1_bars = axes[2].containers[0]
        np.testing.assert_allclose([bar.get_height() for bar in v1_bars], [.007, .006, .001])
    finally:
        _close(figures)


def test_coin_extreme_and_pressure_use_exact_original_event_values():
    # Replacing event balances with the same-day close would give 1.4/1.3, not .55/.25.
    figures = build_figures(*fixture())
    try:
        ax = figures['fig_3.png'].axes[1]
        extreme = next(mark for mark in ax.collections if mark.get_label() == 'V4 maximum USD drawdown')
        pressure = next(mark for mark in ax.collections if mark.get_label() == 'V4 pressure liquidation')
        assert float(extreme.get_offsets()[0, 1]) == .55
        assert float(pressure.get_offsets()[0, 1]) == .25
        assert float(pressure.get_offsets()[0, 0]) == pytest.approx(
            mdates.date2num(pd.Timestamp('2024-10-31T03:00:00Z')))
        assert 'C / C0' in ax.get_ylabel()
    finally:
        _close(figures)


def test_eth_periods_are_separate_with_matching_passive_lines_and_na_sharpe():
    args = fixture()
    pm = args[-1]['period_metrics']
    pm.loc[(pm.asset == 'ETH') & (pm.version == 'V1_USDM') &
           (pm.period == 'IS-Common'), 'sharpe'] = np.nan
    figures = build_figures(*args)
    try:
        axes = figures['fig_4.png'].axes
        expected_labels = {'V1 UM', 'P UM', 'V2 Spot', 'P Spot', 'V3 CM hedge', 'P CM hedge'}
        for ax in axes[:2]:
            assert {line.get_label() for line in ax.lines if line.get_label() in expected_labels} == expected_labels
        assert any(line.get_label() == 'VAL / OOS boundary' for line in axes[1].lines)
        assert max(pd.Timestamp(x) for x in axes[0].lines[0].get_xdata()) < pd.Timestamp('2024-01-01T00:00:00Z')
        assert min(pd.Timestamp(x) for x in axes[1].lines[0].get_xdata()) >= pd.Timestamp('2024-01-01T00:00:00Z')
        assert any('NA' in text.get_text() for text in axes[4].texts)
    finally:
        _close(figures)


def test_absorbed_zero_segment_preserves_undefined_maxdd_without_blocking_figures():
    # A ruined OOS predecessor has no new risk denominator; NA must remain NA.
    args = fixture()
    metrics = args[-1]['period_metrics']
    selected = ((metrics['asset'] == 'BTC') & (metrics['period'] == 'OOS') &
                (metrics['version'] == 'V4_CM_CoinAccum'))
    metrics.loc[selected, 'start_equity'] = 0.
    metrics.loc[selected, ['max_drawdown', 'sharpe']] = np.nan
    figures = build_figures(*args)
    try:
        assert set(figures) == set(EXPECTED_FILES)
        assert metrics.loc[selected, 'max_drawdown'].isna().all()
    finally:
        _close(figures)


def test_render_four_readable_pngs_without_mutating_inputs(tmp_path):
    args = fixture()
    before = deepcopy(args)
    assert render_outputs(*args, tmp_path) == EXPECTED_FILES
    for name in EXPECTED_FILES:
        pixels = mpimg.imread(tmp_path / name)
        assert pixels.shape[0] >= 1000 and pixels.shape[1] >= 1500
        assert (tmp_path / name).stat().st_size > 10_000
    captions = (tmp_path / 'figure_captions.md').read_text()
    assert 'No 69-month continuous' in captions
    assert '74-month' not in captions
    for key, account in args[0].items():
        pd.testing.assert_frame_equal(account['chart_daily'], before[0][key]['chart_daily'])
        pd.testing.assert_frame_equal(account['daily'], before[0][key]['daily'])
        pd.testing.assert_frame_equal(account['pressure_events'], before[0][key]['pressure_events'])
        assert account['metadata'] == before[0][key]['metadata']
    pd.testing.assert_frame_equal(args[3], before[3])
    pd.testing.assert_frame_equal(args[4]['period_metrics'], before[4]['period_metrics'])


@pytest.mark.parametrize('broken', ['missing_account', 'missing_minute_extreme', 'nonfinite_equity',
                                    'duplicate_period', 'negative_initial_coin', 'undefined_dd_with_positive_start'])
def test_invalid_chart_inputs_stop_before_writing(tmp_path, broken):
    args = fixture()
    key = 'BTC__Post-IS__UM__V4_CM_CoinAccum__main_17__continuous'
    if broken == 'missing_account':
        del args[0][key]
    elif broken == 'missing_minute_extreme':
        args[0][key]['chart_daily'].drop(columns='equity_drawdown_minute_min_time', inplace=True)
    elif broken == 'nonfinite_equity':
        args[0][key]['chart_daily'].iloc[0, 0] = np.inf
    elif broken == 'duplicate_period':
        frame = args[4]['period_metrics']
        args[4]['period_metrics'] = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif broken == 'negative_initial_coin':
        args[0][key]['metadata']['initial_coin'] = -1.
    else:
        metrics = args[4]['period_metrics']
        metrics.loc[(metrics['asset'] == 'BTC') & (metrics['period'] == 'OOS') &
                    (metrics['version'] == 'V4_CM_CoinAccum'), 'max_drawdown'] = np.nan
    with pytest.raises(ValueError):
        render_outputs(*args, tmp_path)
    assert not list(tmp_path.iterdir())


def test_existing_artifact_is_never_overwritten(tmp_path):
    path = tmp_path / 'fig_1.png'
    path.write_bytes(b'preserved')
    with pytest.raises(FileExistsError):
        render_outputs(*fixture(), tmp_path)
    assert path.read_bytes() == b'preserved'

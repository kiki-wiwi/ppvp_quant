"""Only invented minute bars: causal feature and synthetic-fixture checks."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest


def frame(minutes, start='2020-03-01', price=100.):
    index = pd.date_range(start, periods=minutes, freq='min', tz='UTC').as_unit('ns')
    return pd.DataFrame({'open': price, 'high': price + 1, 'low': price - 1,
                         'close': price, 'volume': 10., 'bad': False,
                         'archive_gap': False}, index=index)


def context(data):
    from dip_limit_study.scope import make_context
    data.attrs['provenance'] = 'synthetic'
    return make_context(data)


def features(data):
    from dip_limit_study.features import build_features
    return build_features(context(data))


def signals(data, built=None):
    from dip_limit_study.features import build_signals
    return build_signals(context(data), built)


def warm_frame(h='2020-03-01 00:00'):
    """55 days of rising closes; the final hour falls six units, prior ATR=2."""
    h = pd.Timestamp(h, tz='UTC')
    data = frame(55 * 1440 + 60, str((h - pd.Timedelta(days=55)).tz_localize(None)))
    price = 100. + np.arange(len(data)) // 60 * .1
    data.loc[:, 'open'] = price
    data.loc[:, 'high'] = price + 1
    data.loc[:, 'low'] = price - 1
    data.loc[:, 'close'] = price
    data.loc[h:, 'close'] = 226.
    data.loc[h:, 'low'] = 225.
    return data, h


def test_closed_utc_buckets_exclude_bad_minutes_and_unfinished_suffix():
    data = frame(4 * 60 + 37, '2020-03-01 00:17')
    data.iloc[0, data.columns.get_indexer(['high', 'low', 'close', 'volume', 'bad'])] = [999, 1, 999, 999, True]
    built = features(data)
    assert list(built.hours.index.hour) == [0, 1, 2, 3]
    first = built.hours.iloc[0]
    assert first.n_valid == 42 and first.high == 101 and first.low == 99
    assert first.volume == 420
    assert built.hours.index.unit == 'ns' and str(built.hours.index.tz) == 'UTC'
    assert list(built.references.loc['4h'].index) == [pd.Timestamp('2020-03-01 04:00', tz='UTC')]


def test_wilder_state_uses_valid_hours_and_does_not_decay_through_empty_hours():
    data = frame(17 * 60)
    data.loc[data.index[5 * 60:6 * 60], 'bad'] = True
    data.loc[data.index[2 * 60:3 * 60 - 1], 'bad'] = True
    data.loc[data.index[15 * 60:16 * 60], ['open', 'high', 'low', 'close']] = [109, 110, 108, 109]
    data.loc[data.index[16 * 60:], 'bad'] = True
    out = features(data).hours
    assert out.atr_after_hour.iloc[:14].isna().all()
    assert out.atr_after_hour.iloc[14] == 2.
    assert out.atr_after_hour.iloc[15] == pytest.approx(36 / 14)
    assert out.atr_after_hour.iloc[16] == out.atr_after_hour.iloc[15]


def test_signal_uses_prior_atr_and_mean_and_freezes_rounded_order_prices():
    data, h = warm_frame()
    built = features(data)
    row = signals(data, built).loc[h + pd.Timedelta(hours=1)]
    assert row.eligible and row.atr == 2. and row.previous_close == pytest.approx(231.9)
    assert row.target == pytest.approx(230.95) and row.eref == 226.
    assert row.sl == 222. and row.limit == 225.
    assert row.deadline == h + pd.Timedelta(hours=25)
    assert built.hours.loc[h, 'atr_after_hour'] > row.atr
    # A larger signal-hour range must not enlarge its own ATR denominator.
    data.loc[h:, 'high'] = 400.
    wider = signals(data).loc[row.decision_time]
    assert wider.atr == row.atr and wider.target == row.target and wider.sl == row.sl


def test_daily_and_four_hour_candles_closing_at_decision_cannot_rewrite_background():
    data, h = warm_frame('2020-03-01 23:00')
    data.loc[h:, ['close', 'low']] = [116., 115.]
    built = features(data)
    row = signals(data, built).loc[h + pd.Timedelta(hours=1)]
    assert row.eligible and row.daily_trend and row.four_hour_trend
    assert row.daily_reference_time == pd.Timestamp('2020-03-01 00:00', tz='UTC')
    assert row.four_hour_reference_time == pd.Timestamp('2020-03-01 20:00', tz='UTC')
    for frequency in ('1D', '4h'):
        assert not built.references.loc[(frequency, row.decision_time), 'trend']


@pytest.mark.parametrize('frequency', ['1D', '4h'])
def test_both_higher_timeframes_must_have_bullish_trend(frequency):
    data, h = warm_frame()
    built = features(data)
    # Exercise a valid precomputed feature snapshot with one bearish reference.
    built.references.loc[(frequency, h), 'trend'] = False
    row = signals(data, built).loc[h + pd.Timedelta(hours=1)]
    assert not row.eligible and row.skip_reason == 'not_bullish_trend'


@pytest.mark.parametrize('where', ['current', 'oldest_previous'])
def test_every_minute_of_signal_and_twenty_previous_hours_is_required(where):
    data, h = warm_frame()
    when = h if where == 'current' else h - pd.Timedelta(hours=20)
    data.loc[when, 'bad'] = True
    row = signals(data).loc[h + pd.Timedelta(hours=1)]
    assert not row.eligible
    assert row.skip_reason == ('incomplete_signal_hour' if where == 'current' else 'incomplete_mean_history')


def test_fixed_sma_grid_does_not_replace_empty_daily_bar_with_older_close():
    data, h = warm_frame()
    gap = h - pd.Timedelta(days=30)
    data.loc[gap:gap + pd.Timedelta(days=1) - pd.Timedelta(minutes=1), 'bad'] = True
    built = features(data)
    assert np.isnan(built.references.loc[('1D', h), 'sma50'])
    assert not signals(data, built).loc[h + pd.Timedelta(hours=1), 'eligible']
    # A nonempty daily bucket can provide its valid close, however sparse it is.
    data.loc[gap + pd.Timedelta(hours=23, minutes=59), 'bad'] = False
    assert signals(data).loc[h + pd.Timedelta(hours=1), 'eligible']


def test_decision_is_identical_without_entry_minute_or_with_bad_extreme_entry():
    data, h = warm_frame()
    before = signals(data)
    d = h + pd.Timedelta(hours=1)
    assert before.loc[d, 'eligible']
    extra = frame(1, str(d.tz_localize(None)), 1_000_000.)
    extra.loc[:, 'bad'] = True
    pd.testing.assert_frame_equal(before, signals(pd.concat([data, extra])))


def test_archive_gap_is_disclosed_without_filtering_a_valid_signal():
    data, h = warm_frame()
    before = signals(data).loc[h + pd.Timedelta(hours=1)]
    data.loc[data.index[1], 'archive_gap'] = True
    after = signals(data).loc[h + pd.Timedelta(hours=1)]
    assert before.eligible and after.eligible
    assert not before.signal_lookback_has_archive_gap and after.signal_lookback_has_archive_gap
    assert before.atr == after.atr and before.target == after.target


def test_prefix_invariance_retains_utc_and_nanosecond_resolution():
    data, _ = warm_frame()
    prefix = data.iloc[:-23].copy()
    prefix.index = prefix.index.tz_convert('Asia/Shanghai').as_unit('us')
    full = features(data)
    short = features(prefix)
    pd.testing.assert_frame_equal(short.hours, full.hours.loc[short.hours.index])
    pd.testing.assert_frame_equal(short.references, full.references.loc[short.references.index])
    prefix_signals = signals(prefix)
    pd.testing.assert_frame_equal(prefix_signals, signals(data).loc[prefix_signals.index])


@pytest.mark.parametrize('h, permitted', [('2020-02-29 23:00', False),
    ('2020-03-01 00:00', True), ('2023-12-30 22:00', True), ('2023-12-30 23:00', False)])
def test_main_and_last_decision_limits_do_not_require_future_prices(h, permitted):
    data, opened = warm_frame(h)
    row = signals(data).loc[opened + pd.Timedelta(hours=1)]
    assert bool(row.eligible) is permitted


def test_short_prefix_is_empty_and_foreign_source_fails_before_feature_work():
    from dip_limit_study.features import build_features, build_signals
    from dip_limit_study.scope import RealDataLockedError
    ctx = context(frame(1))
    assert build_features(ctx).hours.empty and build_signals(ctx).empty
    for function in (build_features, build_signals):
        with pytest.raises(RealDataLockedError):
            function(SimpleNamespace(source='real', symbol='BTCUSDT'))


def test_full_synthetic_fixture_has_all_46_months_of_fillable_invented_signals():
    from dip_limit_study.synthetic import study_context
    from dip_limit_study.features import build_signals
    ctx = study_context()
    assert ctx.source == 'synthetic' and ctx.symbol == 'BTCUSDT'
    assert ctx.index[0] == pd.Timestamp('2020-01-01', tz='UTC')
    assert ctx.index[-1] == pd.Timestamp('2023-12-31 23:59', tz='UTC')
    assert len(ctx) == 2_103_840
    eligible = build_signals(ctx).query('eligible')
    assert len(eligible) >= 150
    assert eligible.index.tz_localize(None).to_period('M').nunique() == 46
    # Independent event intention: activation stays passive and both fill thresholds
    # are touched before the fixed stop in the same two-minute invented path.
    for row in eligible.itertuples():
        i = ctx.index.get_loc(row.decision_time)
        delta = np.ceil(max(.1, .0002 * row.limit) / .1) * .1
        assert ctx.O[i] > row.limit and ctx.O[i + 1] > row.limit
        assert ctx.L[i + 1] <= row.limit - delta
        assert ctx.L[i + 1] > row.sl


@pytest.mark.parametrize('eref, stop, limit', [(103.3, 99.3, 102.3),
    (103.29999999999999, 99.2, 102.2), (103.301, 99.3, 102.3)])
def test_decimal_grid_floor_keeps_exact_ticks_without_promoting_nearby_lower_prices(eref, stop, limit):
    data, h = warm_frame()
    data.loc[h:, 'close'] = eref
    data.loc[h:, 'low'] = eref - 1.
    row = signals(data).loc[h + pd.Timedelta(hours=1)]
    # Prior ATR is exactly 2. For 103.3, the decimal offsets give 99.3 and
    # 102.3 exactly; a representable value strictly below must still floor down.
    assert row.eligible and row.atr == 2.
    assert row.sl == stop
    assert row.limit == limit

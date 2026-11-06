"""Synthetic causal-channel checks; no market input or old research locks."""
import importlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from channel_random_study.scope import make_context


H = pd.Timedelta(hours=1)
T = pd.Timestamp('2020-03-01 03:00', tz='UTC')


def frame(start='2020-02-23', end='2020-03-02'):
    idx = pd.date_range(start, end, freq='min', inclusive='left', tz='UTC').as_unit('ns')
    result = pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=1.,
                               bad=False, archive_gap=False), index=idx)
    result.attrs['provenance'] = 'synthetic'
    return result


def breakout_frame():
    result = frame()
    # No volume during the preceding 20 hours or the signal; reference volume remains positive.
    result.loc[T-20*H:T+H-pd.Timedelta(minutes=1), 'volume'] = 0.
    result.loc[T:T+H-pd.Timedelta(minutes=1), ['high','close']] = [104.,103.]
    # A high in the current 4H bucket must not enter U frozen at h.
    result.loc[T-2*H:T-H-pd.Timedelta(minutes=1), 'high'] = 110.
    return result


def test_unfiltered_parent_uses_h_reference_and_current_atr_but_previous_prevol():
    f = importlib.import_module('channel_random_study.features')
    ctx = make_context(breakout_frame())
    built = f.build_features(ctx)
    opportunities = f.build_opportunities(ctx, built)
    row = opportunities.loc[T+H]
    assert row.eligible and row.is_parent
    assert row.reference_time == pd.Timestamp('2020-03-01', tz='UTC')
    assert row.boundary == 101.
    assert row.eref == 103. and row.previous_close == 100.
    assert row.atr == built.hours.loc[T, 'atr']
    assert row.prevol == built.hours.loc[T-H, 'atr'] / 100.
    assert row.atr != row.previous_atr
    assert row.sl == row.eref - 2*row.atr
    assert row.skip_reason == ''
    assert opportunities.index.name == 'entry_time'
    assert str(opportunities.index.dtype) == 'datetime64[ns, UTC]'
    assert not opportunities.loc[opportunities.is_parent, 'eligible'].eq(False).any()
    assert not set(('POC','VAL','VAH','compression_q','baseline_volume')) & set(opportunities.columns)


def test_common_eligibility_keeps_nonparents_and_does_not_inspect_bad_entry():
    f = importlib.import_module('channel_random_study.features')
    data = breakout_frame()
    data.loc[T+H, 'bad'] = True
    result = f.build_opportunities(make_context(data))
    assert result.loc[T+H, 'is_parent']
    assert result.loc[T+H, 'eligible']
    ordinary = result.loc[pd.Timestamp('2020-03-01 01:00', tz='UTC')]
    assert ordinary.eligible and not ordinary.is_parent


def test_full_hour_quality_and_calendar_reference_holes_are_not_compressed_away():
    f = importlib.import_module('channel_random_study.features')
    data = breakout_frame()
    data.loc[T-8*H, 'bad'] = True
    row = f.build_opportunities(make_context(data)).loc[T+H]
    assert not row.eligible and row.skip_reason == 'incomplete_hour_history'
    data = breakout_frame()
    missing_start = pd.Timestamp('2020-02-27', tz='UTC')
    data.loc[missing_start:missing_start+4*H-pd.Timedelta(minutes=1), 'bad'] = True
    row = f.build_opportunities(make_context(data)).loc[T+H]
    assert not row.eligible and row.skip_reason == 'invalid_reference'
    data = breakout_frame()
    data.loc[missing_start, ['bad','archive_gap']] = True
    row = f.build_opportunities(make_context(data)).loc[T+H]
    assert row.eligible and row.window_has_archive_gap and row.atr_has_archive_gap
    assert row.signal_lookback_has_archive_gap


def test_wilder_atr_skips_empty_hours_but_updates_partial_hours():
    f = importlib.import_module('channel_random_study.features')
    data = frame('2020-03-01', '2020-03-02')
    empty = pd.Timestamp('2020-03-01 14:00', tz='UTC')
    partial = empty+H
    data.loc[empty:partial+H-pd.Timedelta(minutes=1), 'bad'] = True
    data.loc[partial, ['bad','high']] = [False,103.]
    hours = f.build_features(make_context(data)).hours
    assert np.isnan(hours.iloc[12].atr)
    assert hours.iloc[13].atr == 2.
    assert hours.loc[empty, 'n_valid'] == 0 and hours.loc[empty, 'atr'] == 2.
    assert hours.loc[partial, 'n_valid'] == 1
    assert hours.loc[partial, 'atr'] == pytest.approx(30/14)


def test_completed_feature_and_opportunity_prefix_is_unchanged_by_future_minutes():
    f = importlib.import_module('channel_random_study.features')
    data = breakout_frame()
    cut = T+H
    prefix = data.loc[data.index < cut].copy()
    future = data.copy()
    future.loc[future.index >= cut, ['open','high','low','close','volume']] = [1000.,1010.,990.,1000.,9000.]
    a = f.build_features(make_context(prefix))
    b = f.build_features(make_context(future))
    assert_frame_equal(a.hours, b.hours.loc[a.hours.index])
    assert_frame_equal(a.references, b.references.loc[a.references.index])
    assert_frame_equal(f.build_opportunities(make_context(prefix), a),
                       f.build_opportunities(make_context(future), b).loc[:cut])


@pytest.mark.parametrize('value,want', [
    (np.nextafter(.0025, 0),0), (.0025,1), (.005,2), (.01,3), (.02,4),
    (0.,5), (-.01,5), (np.nan,5), (np.inf,5),
])
def test_bins_use_prior_hour_direct_float64_boundaries_and_missing_bin_does_not_filter(value, want):
    f = importlib.import_module('channel_random_study.features')
    ctx = make_context(breakout_frame())
    built = f.build_features(ctx)
    built.hours.loc[T-H, 'atr'] = value * 100.
    row = f.build_opportunities(ctx, built).loc[T+H]
    assert row.vol_bin == want
    assert row.eligible and row.is_parent


def test_signal_hour_start_and_last_entry_boundaries():
    f = importlib.import_module('channel_random_study.features')
    early = f.build_opportunities(make_context(frame()))
    assert early.loc[pd.Timestamp('2020-03-01', tz='UTC'), 'skip_reason'] == 'before_main_start'
    assert early.loc[pd.Timestamp('2020-03-01 01:00', tz='UTC'), 'eligible']
    late = f.build_opportunities(make_context(frame('2023-12-22', '2023-12-30')))
    assert late.loc[pd.Timestamp('2023-12-28 23:00', tz='UTC'), 'eligible']
    assert late.loc[pd.Timestamp('2023-12-29', tz='UTC'), 'skip_reason'] == 'after_last_entry'

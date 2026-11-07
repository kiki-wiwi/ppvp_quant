"""Synthetic tests for daily availability, risk budgets, and market-free fixtures."""

import numpy as np
import pandas as pd
import pytest

from trend_hold_study import config, scope
from trend_hold_study.features import build_features
from trend_hold_study.synthetic import study_context

DAY = pd.Timedelta(days=1)
MINUTE = pd.Timedelta(minutes=1)


def make_daily_fixture(closes, *, start=config.IS_START, final_minutes=1, alter=None):
    index = pd.date_range(start, periods=len(closes) * 1440 + final_minutes,
                          freq='min').as_unit('us')
    prices = np.concatenate([np.repeat(closes, 1440), np.full(final_minutes, 100_000.)])
    frame = pd.DataFrame({name: prices for name in ('open', 'high', 'low', 'close')},
                         index=index)
    frame['volume'] = 1.
    frame['bad'] = False
    frame['archive_gap'] = False
    frame.attrs['provenance'] = 'synthetic'
    if alter is not None:
        alter(frame)
    return scope.make_context(frame)


def test_daily_windows_use_only_completed_prior_closes():
    # The large change on day 39 belongs to the SMA but not the first 20 returns.
    closes = np.full(63, 100.)
    closes[38] = 1000.
    closes[59] = 120.
    closes[60:] = [300., 10., 80.]
    one = build_features(make_daily_fixture(closes[:60]))
    full = build_features(make_daily_fixture(closes))
    pd.testing.assert_frame_equal(one, full.loc[one.index], check_freq=False)
    assert full.index[0] == config.MAIN_START
    assert full.index.unit == 'ns'
    first = full.iloc[0]
    assert first.eref == 120.
    assert first.sma60 == pytest.approx((58 * 100. + 1000. + 120.) / 60.)
    # 19 zero returns and one +20% return, giving sample SD .2/sqrt(20).
    assert first.sigma == pytest.approx(.2 * np.sqrt(365. / 20.))
    assert bool(first.g)
    assert first.budget == pytest.approx(np.sqrt(20. / 365.))
    assert full.iloc[1].eref == 300.
    assert not bool(full.iloc[2].g)
    assert full.budget_update.to_list() == [True, True, False, False]
    assert full.iloc[1].budget != first.budget
    assert full.iloc[1:].budget.nunique() == 1


def test_zero_sigma_and_equal_close_turn_trend_off():
    features = build_features(make_daily_fixture(np.full(60, 100.)))
    assert len(features) == 1
    first = features.iloc[0]
    assert first.eref == first.sma60 == 100.
    assert first.sigma == 0.
    assert first.budget == 1.
    assert not bool(first.g)


def test_partial_valid_day_uses_its_last_valid_close_and_reports_gap():

    def alter(frame):
        last = config.MAIN_START - MINUTE
        frame.loc[last - MINUTE, 'close'] = 125.
        frame.loc[last - MINUTE, 'high'] = 125.
        frame.loc[last, 'close'] = 900.
        frame.loc[last, 'bad'] = True
        frame.loc[last, 'archive_gap'] = True

    first = build_features(make_daily_fixture(np.full(60, 100.), alter=alter)).iloc[0]
    assert first.eref == 125.
    assert first.n_valid_prior == 1439
    assert bool(first.has_archive_gap)


def test_complete_day_with_no_valid_minutes_stops():

    def alter(frame):
        frame.loc[(frame.index >= config.MAIN_START - DAY)
                  & (frame.index < config.MAIN_START), 'bad'] = True

    with pytest.raises(ValueError, match='no valid'):
        build_features(make_daily_fixture(np.full(60, 100.), alter=alter))


def test_partial_tail_cannot_supply_a_close_or_decision_after_its_end():

    def alter(frame):
        frame.loc[frame.index >= config.MAIN_START, 'bad'] = True

    complete_warmup = build_features(make_daily_fixture(np.full(60, 100.), final_minutes=0))
    partial_execution = build_features(make_daily_fixture(np.full(60, 100.), final_minutes=719, alter=alter))
    assert complete_warmup.empty
    assert partial_execution.index.to_list() == [config.MAIN_START]
    assert partial_execution.iloc[0].eref == 100.


def test_main_decision_rejects_insufficient_complete_warmup():
    with pytest.raises(ValueError, match='60 completed'):
        build_features(make_daily_fixture(np.full(59, 100.), start=config.IS_START + DAY))


def test_study_fixture_has_full_calendar_and_changing_regimes():
    ctx = study_context()
    features = build_features(ctx)
    assert len(ctx) == 2_103_840 and ctx.source == 'synthetic'
    assert ctx.index[0] == config.IS_START and ctx.index[-1] == config.IS_END - MINUTE
    assert len(features) == 1401 and features.index[-1] == config.IS_END - DAY
    assert features.g.any() and not features.g.all()
    assert features.sigma.nunique() > 1
    assert 0 < features.budget.min() < 1.
    assert ctx.bad.any() and ctx.archive_gap.any()
    assert np.isfinite(ctx.funding_rate).sum() == 4383
    assert not ctx.bad[-1]

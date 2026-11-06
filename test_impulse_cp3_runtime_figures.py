"""Synthetic boundary/draw-order tests; never opens market files."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from impulse_study import cp3_runtime as R, engine as E, synthetic as Y, features as F, versions as VS


def test_funding_predicates_and_rounding_happen_at_entry():
    times = pd.date_range('2020-01-01', '2024-01-01', freq='8h', inclusive='left', tz='UTC')
    frame = pd.DataFrame({'funding_rate': np.full(len(times), .0001)}, index=times + pd.Timedelta(milliseconds=2))
    with patch.object(R.pd, 'read_parquet', return_value=frame) as read:
        got = R.load_funding_is()
    filters = read.call_args.kwargs['filters']
    assert ('funding_time', '<', pd.Timestamp('2024-01-01', tz='UTC')) in filters
    assert ('funding_time', '>=', pd.Timestamp('2020-01-01', tz='UTC')) in filters
    assert len(got) == 4383 and (got.index.second == 0).all()
    assert read.call_args.args[0].name == 'BTCUSDT_funding.parquet'


def test_loader_refuses_before_unlock_without_any_io(monkeypatch):
    monkeypatch.setattr(E, 'REAL_DATA_UNLOCKED', False)
    with patch.object(R.D, '_manifest', side_effect=AssertionError('must not read')), pytest.raises(E.EngineLockedError):
        R.load_real_context()


def test_frozen_table_and_key_match_without_market_reads():
    assert len(R.frozen_hashes()) == 27
    assert VS.frozen_key() == 'fc0d37b9d198ff85'


def test_figure_draws_shared_across_columns_match_direct_reference():
    from impulse_study.cp3_figures import shared_month_intervals
    times = pd.date_range('2020-03-01', periods=20, freq='MS', tz='UTC')
    values = np.column_stack([np.arange(20.), 2 * np.arange(20.) - 3])
    got = shared_month_intervals(times, values, VS.SEED_BASE + 1100)
    draws = np.random.default_rng(VS.SEED_BASE + 1100).integers(0, 20, size=(2000, 20))
    expected = np.percentile(values[draws].mean(axis=1), [2.5, 97.5], axis=0)
    np.testing.assert_allclose(got.lo, expected[0], rtol=1e-12)
    np.testing.assert_allclose(got.hi, expected[1], rtol=1e-12)
    assert (got.discarded == 0).all()


def test_lifecycle_first_minute_and_is_end_do_not_read_beyond_grid():
    from impulse_study.cp3_figures import lifecycle_rows
    O,H,L,C,V = Y.flat_bars(360, spread=.01)
    C[311] = O[311] * 1.01
    g = Y.grid_from_arrays('2023-12-31 18:00',O,H,L,C,V)
    ctx = F.make_context(g, np.zeros(len(g),bool), 'synthetic', main_start=g.index[300], last_trigger=g.index[329])
    tr = pd.DataFrame({'trigger_idx':[310,329], 'trigger_time':ctx.index[[310,329]],
                       'entry_time':ctx.index[[311,330]], 'exit_time':ctx.index[[326,345]]})
    got = lifecycle_rows(ctx,tr,1)
    assert got[0,0] == pytest.approx(100)
    assert np.isnan(got[1,30:]).all()
    assert np.isfinite(got[1,:30]).all()


def test_category_intervals_degenerate_and_empty_are_explicit():
    from impulse_study.cp3_figures import shared_month_intervals
    times = pd.date_range('2021-01-01', periods=11, freq='MS', tz='UTC')
    got = shared_month_intervals(times, np.ones((11,2)), 77)
    assert got.lo.isna().all() and (got['mean'] == 1).all()
    got = shared_month_intervals(pd.DatetimeIndex([],tz='UTC'), np.empty((0,3)), 77)
    assert got['mean'].isna().all() and (got.n == 0).all()

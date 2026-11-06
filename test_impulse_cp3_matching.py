"""Synthetic checks for frozen matching order, relaxation and event pairing.

No real-data loader is imported or called. Test changes which would fail these
checks include altered q population/edges, opposite-direction leakage, RNG use
for missing rows, changed K/seeds, cost-specific redraws and altered stop scale.
"""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from impulse_study import engine as E, features as F, synthetic as SY, versions as VS
from impulse_study import cp3_matching as M
from impulse_study.scope import ScopeError


def controlled_context(candidates, *, n=3000, start='2020-03-01', atr=None):
    grid = SY.grid_from_arrays(start, *SY.flat_bars(n, spread=.001))
    ctx = SY.synthetic_context(grid, grid.index[0], grid.index[-31], box_ok=np.ones(n, bool))
    up1 = np.ones(n, bool)
    up1[np.asarray(candidates, dtype=int)] = False
    condition = {'up1': up1, 'up2': np.zeros(n, bool), 'dn1': np.zeros(n, bool), 'dn2': np.zeros(n, bool)}
    frame = ctx.frame.copy(deep=True)
    frame['contaminated'] = False
    frame['sigma7d'] = .001
    atr_dict = dict(ctx.atr)
    atr_dict['atr_at'] = np.full(n, 10.) if atr is None else np.asarray(atr, float).copy()
    return replace(ctx, cond={'pct': condition, 'z': {k: x.copy() for k, x in condition.items()}},
                   frame=frame, atr=atr_dict, base_med=np.full(n, 150.))


def ledger(ctx, indices, *, side=1, stop_atr=2.5):
    i = np.asarray(indices, dtype=np.int64)
    return pd.DataFrame({
        'trigger_idx': i, 'trigger_time': ctx.index[i], 'entry_time': ctx.index[i + 1],
        'exit_time': ctx.index[i + 16], 'side': np.full(len(i), side, dtype=np.int64),
        'e_ref': ctx.C[i], 'sl': ctx.C[i] - side * stop_atr * ctx.atr['atr_at'][i],
        'atr15': ctx.atr['atr_at'][i], 'net_bps': np.arange(len(i), dtype=float),
    })


@pytest.mark.parametrize('pool,other_band,expected', [
    (np.arange(730, 750), False, 'month_hour_atr'),
    (np.arange(780, 800), False, 'month_6hour_atr'),
    (np.arange(60, 80), False, 'month_atr'),
    (np.arange(730, 750), True, 'month_ge20'),
    (np.arange(730, 749), False, 'month_1to19'),
    (np.array([], dtype=int), False, 'missing_month'),
])
def test_six_exclusive_relaxation_classes_and_distinct_pool_threshold(pool, other_band, expected):
    atr = np.full(3000, 10.)
    if other_band:
        atr[pool] = 20.
    ctx = controlled_context(pool, atr=atr)
    design = M.build_match_design(ctx, VS.VERSIONS['B1_pure_long'], ledger(ctx, [720]))
    assert design.records.loc[0, 'match_stratum'] == expected
    assert design.records.loc[0, 'candidate_count'] == len(pool)
    assert design.relaxation_counts[expected] == 1
    assert sum(design.relaxation_counts.values()) == 1
    if len(pool):
        assert np.isin(design.sample_indices[0], pool).all()
        # Sampling has replacement: 100 draws from 19/20 distinct minutes.
        assert design.sample_indices.shape == (1, 100)
    else:
        assert (design.sample_indices == -1).all()


def test_missing_month_consumes_no_rng_and_actuals_are_sorted_by_trigger():
    pool = np.arange(2170, 2190)
    ctx = controlled_context(pool, start='2020-03-31')
    spec = VS.VERSIONS['B1_pure_long']
    design = M.build_match_design(ctx, spec, ledger(ctx, [2160, 720]))
    seed = VS.SEED_BASE + 100 + list(VS.VERSIONS).index(spec.name)
    expected = pool[np.random.default_rng(seed).integers(0, len(pool), size=100)]
    assert design.records.trigger_idx.tolist() == [720, 2160]
    assert (design.sample_indices[0] == -1).all()
    np.testing.assert_array_equal(design.sample_indices[1], expected)
    assert design.records.trigger_month.tolist() == ['2020-03', '2020-04']


def test_quantiles_use_all_nonbad_ready_main_minutes_not_candidate_pool():
    atr = np.repeat([10., 20., 30.], 1000)
    ctx = controlled_context([730, 1730, 2730], atr=atr)
    design = M.build_match_design(ctx, VS.VERSIONS['B1_pure_long'], ledger(ctx, [720, 1720, 2720]))
    expected = np.quantile(atr[ctx.window] / ctx.C[ctx.window], [1/3, 2/3])
    np.testing.assert_array_equal([design.q1, design.q2], expected)
    # Equality belongs to the lower band (a <= q1, then a <= q2).
    expected_bands = np.searchsorted(expected, atr[[720, 1720, 2720]] / 10000, side='left') + 1
    np.testing.assert_array_equal(design.records.trigger_atr_band, expected_bands)


def test_candidates_exclude_opposite_signal_bad_entry_and_undefined_features():
    pool = np.arange(730, 760)
    ctx = controlled_context(pool)
    cond = {rule: {key: arr.copy() for key, arr in row.items()} for rule, row in ctx.cond.items()}
    cond['pct']['dn2'][730] = True
    frame = ctx.frame.copy(deep=True)
    frame.loc[ctx.index[731], 'contaminated'] = True
    atr = {**ctx.atr, 'atr_at': ctx.atr['atr_at'].copy()}
    atr['atr_at'][732] = 0
    atr['atr_at'][733] = np.nan
    base = ctx.base_med.copy()
    base[734] = np.nan
    bad = ctx.bad.copy()
    bad[736] = True  # candidate 735's entry is invalid
    derived = replace(ctx, cond=cond, frame=frame, atr=atr, base_med=base, bad=bad)
    design = M.build_match_design(derived, VS.VERSIONS['B1_pure_long'], ledger(derived, [720]))
    for excluded in [730, 731, 732, 733, 734, 735]:
        assert excluded not in design.candidate_indices
    assert 759 in design.candidate_indices


def test_box_and_z_candidates_apply_their_own_eligibility():
    pool = np.arange(730, 760)
    ctx = controlled_context(pool)
    box = ctx.box_ok.copy()
    box[730] = False
    frame = ctx.frame.copy(deep=True)
    frame.loc[ctx.index[731], 'sigma7d'] = np.nan
    derived = replace(ctx, box_ok=box, frame=frame)
    a = M.build_match_design(derived, VS.VERSIONS['A5_box_origin_z'], ledger(derived, [720]))
    b = M.build_match_design(derived, VS.VERSIONS['B1_pure_long'], ledger(derived, [720]))
    assert 730 not in a.candidate_indices and 731 not in a.candidate_indices
    assert 730 in b.candidate_indices and 731 in b.candidate_indices
    assert ctx.box_ok[730] and np.isfinite(ctx.frame.loc[ctx.index[731], 'sigma7d'])


def test_all_costs_share_draws_and_use_literal_ledger_stop_distance():
    # Handwritten original stop distance is deliberately not 2.5 ATR.
    ctx = controlled_context(np.arange(730, 755))
    spec = VS.VERSIONS['B1_pure_long']
    main = ledger(ctx, [720], side=-1, stop_atr=2.50000003)
    all_costs = {}
    for cost_name, cost in VS.COSTS.items():
        tr = main.copy()
        event = E.simulate_events(ctx, np.array([720]), np.array([-1.]), tr.sl.to_numpy(),
                                  tr.e_ref.to_numpy(), False, cost)
        tr['net_bps'] = event.net_bps.to_numpy()
        all_costs[cost_name] = tr
    original_C = ctx.C.copy()
    original_conditions = {key: arr.copy() for key, arr in ctx.cond['pct'].items()}
    result = M.run_matched_controls(ctx, spec, all_costs)
    d = abs(main.e_ref.iloc[0] - main.sl.iloc[0]) / main.atr15.iloc[0]
    sampled = result.sample_indices[0]
    for cost_name, cost in VS.COSTS.items():
        reference = ctx.C[sampled]
        expected = E.simulate_events(ctx, sampled, np.full(100, -1.),
                                      reference + d * ctx.atr['atr_at'][sampled], reference, False, cost)
        np.testing.assert_array_equal(result.random_net_bps[cost_name][0], expected.net_bps.to_numpy())
        assert result.records[f'excess_{cost_name}'].iloc[0] == pytest.approx(
            all_costs[cost_name].net_bps.iloc[0] - expected.net_bps.mean())
    np.testing.assert_array_equal(ctx.C, original_C)
    for key, original in original_conditions.items():
        np.testing.assert_array_equal(ctx.cond['pct'][key], original)
    assert result.balance['matched_actual_count'] == 1
    assert sum(result.balance['utc_hour']['random']) == 100


def test_auxiliary_null_exact_frozen_row_major_indices_and_matched_only_mean():
    random = np.arange(300, dtype=float).reshape(3, 100) / 10
    actual = np.array([5., 15., 25.])
    times = pd.to_datetime(['2020-03-01', '2020-03-02', '2020-04-01'], utc=True)
    v = 7
    got = M.auxiliary_randomization(actual, random, times, v)
    j = np.random.default_rng(VS.SEED_BASE + 900 + v).integers(0, 100, size=(5000, 3))
    expected = random[np.arange(3)[None, :], j].mean(axis=1)
    np.testing.assert_array_equal(got['null_means'], expected)
    assert got['p'] == np.mean(expected >= 15.)
    assert got['mcse'] == pytest.approx(np.sqrt(got['p'] * (1 - got['p']) / 5000))


def test_no_candidates_produces_missing_excess_and_does_not_shrink_records():
    ctx = controlled_context([])
    tr = ledger(ctx, [720])
    result = M.run_matched_controls(ctx, VS.VERSIONS['B1_pure_long'], {name: tr.copy() for name in VS.COSTS})
    assert len(result.records) == 1
    assert np.isnan(result.records.excess_bps.iloc[0])
    assert result.randomization['p'] == 1.
    assert result.randomization['null_means'].size == 0
    assert result.relaxation_counts['missing_month'] == 1


def test_matching_scope_rejects_appendix_and_val_before_simulation():
    ctx = controlled_context([730])
    tr = ledger(ctx, [720])
    invalid = tr.copy()
    invalid.loc[0, 'exit_time'] = pd.Timestamp('2024-01-01 00:01', tz='UTC')
    with pytest.raises(ScopeError):
        M.build_match_design(ctx, VS.VERSIONS['B1_pure_long'], invalid)
    appendix = controlled_context([730], start='2020-01-01')
    with pytest.raises(ScopeError):
        M.build_match_design(appendix, VS.VERSIONS['B1_pure_long'], ledger(appendix, [720]))

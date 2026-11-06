"""Synthetic numerical references for the frozen two-hypothesis procedure."""
import hashlib
import importlib
import importlib.util
import math

import numpy as np
import pandas as pd
import pytest

from channel_random_study.scope import ScopeError

SPEC = importlib.util.find_spec('channel_random_study.statistics')
stats = importlib.import_module('channel_random_study.statistics') if SPEC else None
CALENDAR = pd.date_range('2020-03-01', '2023-12-31', freq='D', tz='UTC')
MONTHS = pd.date_range('2020-03-01', '2023-12-01', freq='MS', tz='UTC')
R_NAMES = [f'R_{j:03d}' for j in range(200)]


def test_approved_statistics_module_exists():
    assert stats is not None, 'the approved statistics module must be implemented'


def ledger(month_count=46, per_month=3, bps=12.):
    entries = pd.DatetimeIndex([m + pd.Timedelta(days=j+1, hours=1)
                               for m in MONTHS[:month_count] for j in range(per_month)], tz='UTC')
    return pd.DataFrame({'entry_time': entries, 'exit_time': entries + pd.Timedelta(hours=2),
                         'net_bps': np.full(len(entries), bps)})


def inputs():
    wave = np.sin(np.arange(1401)*.17)*.00008
    s = pd.Series(.0003 + wave, index=CALENDAR)
    r = pd.DataFrame(.00005 + wave[:, None]*np.linspace(.2, .8, 200)
                     + np.linspace(-.00001, .00001, 200), index=CALENDAR, columns=R_NAMES)
    s.iloc[::7] = 0.
    r.iloc[::7] = 0.
    strata = pd.DataFrame({'month': MONTHS.strftime('%Y-%m'), 'utc_hour': 0,
                           'vol_bin': 0, 'N': 30, 'n': 10})
    counts = pd.DataFrame({'trade_count': 100, 'exposure_fraction': .2}, index=R_NAMES)
    return s, r, ledger(), s*.5, ledger(bps=5.), strata, counts


def test_webb_unequal_month_lengths_and_restricted_null_match_literal_reference():
    sums, days = np.array([.12, -.03, .05]), np.array([31., 30., 28.])
    weights = np.array([[1., -1., 1.], [-1., 1., -1.], [0., 0., 0.]])
    out = stats.webb_test(sums, days, weights, has_trades=True)
    # Independently calculate the registered studentized cluster statistic.
    mean = .14/89
    se = math.sqrt(1.5*sum((a-mean*k)**2 for a, k in zip(sums, days)))/89
    null = []
    for row in weights:
        totals = [float(a*w) for a, w in zip(sums, row)]
        mu = sum(totals)/89
        standard_error = math.sqrt(1.5*sum((a-mu*k)**2 for a, k in zip(totals, days)))/89
        null.append(mu/standard_error if standard_error else 0.)
    assert out['mean'] == pytest.approx(mean)
    assert out['se'] == pytest.approx(se)
    assert out['t'] == pytest.approx(mean/se)
    assert out['p_raw'] == (1+sum(t >= mean/se for t in null))/4
    assert out['zero_bootstrap_se_count'] == 1
    assert out['mcse_p'] == pytest.approx(math.sqrt(out['p_raw']*(1-out['p_raw'])/3))


def test_holm_family_has_two_members_stable_ties_and_no_singleton_shortcut():
    out = stats.holm_two({'S': .02, 'D': .02})
    assert [out[name]['rank'] for name in ('S', 'D')] == [1, 2]
    assert [out[name]['p_holm'] for name in ('S', 'D')] == [.04, .04]
    assert [out[name]['holm_threshold'] for name in ('S', 'D')] == [.025, .05]
    assert stats.holm_two({'S': .026, 'D': 1.})['S']['reject'] is False
    with pytest.raises(ValueError):
        stats.holm_two({'S': .01})
    with pytest.raises(ValueError):
        stats.holm_two({'S': .01, 'D': .02, 'R_000': .03})


def test_webb_degenerate_samples_force_one_and_do_not_replace_zero_se_draws():
    weights = np.array([[1., 1., 1.], [0., 0., 0.]])
    out = stats.webb_test(np.zeros(3), np.ones(3), weights, has_trades=True)
    assert out['p_raw'] == 1.
    assert out['zero_bootstrap_se_count'] == 2
    assert out['t'] is None
    no_trades = stats.webb_test(np.array([1., 2., 4.]), np.ones(3), weights, has_trades=False)
    assert no_trades['p_raw'] == 1.
    assert 'no_trades_in_required_path' in no_trades['forced_p_one_reasons']


def test_month_resampling_matches_explicit_daily_and_entry_ledger_concatenation():
    # Two synthetic months contain 1 and 3 days; R accounts have different paths.
    data = {'counts': np.array([1, 3]), 'sums': np.array([.01, .09]),
            'squares': np.array([.0001, .0035]),
            's_days': [np.array([.01]), np.array([.01, .03, .05])],
            'r_sums': np.array([[.02, .00], [.00, .06]]),
            'trade_counts': np.array([1, 2]), 'trade_sums': np.array([10., 60.])}
    draws = np.array([[0, 1], [1, 1], [0, 0]])
    out = stats.month_intervals(data, draws)
    samples = [np.array([.01, .01, .03, .05]), np.array([.01, .03, .05]*2), np.array([.01, .01])]
    sharpes = [math.sqrt(365)*x.mean()/x.std(ddof=1) for x in samples[:2]]
    assert out.loc['S_daily_mean', 'lower'] == pytest.approx(.01075)
    assert out.loc['S_daily_mean', 'upper'] == pytest.approx(.02975)
    assert out.loc['Rbar_daily_mean', 'lower'] == pytest.approx(.01)
    assert out.loc['D_daily_mean', 'upper'] == pytest.approx(.01975)
    assert out.loc['S_sharpe', 'lower'] == pytest.approx(np.quantile(sharpes, .025, method='linear'))
    assert out.loc['S_sharpe', 'undefined_resamples'] == 1
    assert out.loc['S_trade_net_bps', 'upper'] == pytest.approx(np.quantile([70/3, 30, 10], .975, method='linear'))


def test_empty_month_and_constant_returns_only_discard_the_undefined_estimand():
    data = {'counts': np.array([31, 30]), 'sums': np.zeros(2), 'squares': np.zeros(2),
            's_days': [np.zeros(31), np.zeros(30)],
            'r_sums': np.zeros((2, 2)), 'trade_counts': np.array([1, 0]), 'trade_sums': np.array([-5., 0.])}
    out = stats.month_intervals(data, np.array([[0, 1], [1, 1], [0, 0]]))
    assert out.loc['S_trade_net_bps', 'undefined_resamples'] == 1
    assert out.loc['S_trade_net_bps', 'fraction_below_zero'] == 1.
    assert out.loc['S_sharpe', 'undefined_resamples'] == 3
    assert out.loc['D_daily_mean', 'undefined_resamples'] == 0


def test_best_three_entry_months_ties_and_empty_second_half():
    t = ledger(month_count=4, per_month=1)
    t['net_bps'] = [10., 10., 10., -1.]
    out = stats.trade_diagnostics(t)
    assert out['removed_best_months'] == ['2020-03', '2020-04', '2020-05']
    assert out['remaining_trade_mean_bps'] == -1.
    assert out['half_trade_mean_bps'] == [7.25, None]


def test_gate_precedence_and_cost_only_failure():
    gates = {f'U{i}': {'passed': True} for i in range(1, 8)}
    assert stats.candidate_status(gates) == 'is_exploratory_candidate'
    gates['U6']['passed'] = False
    assert stats.candidate_status(gates) == 'cost_sensitive_not_candidate'
    gates['U3']['passed'] = False
    assert stats.candidate_status(gates) == 'no_evidence_beyond_random'
    gates['U7']['passed'] = False
    assert stats.candidate_status(gates) == 'insufficient_random_identifiability_or_precision'
    gates['U1']['passed'] = False
    assert stats.candidate_status(gates) == 'insufficient_sample_evidence'


@pytest.mark.parametrize('mutation', ['missing_day', 'reverse_days', 'naive', 'nonfinite',
                                     'nonpositive_equity', 'reverse_paths', 'missing_path', 'n_gt_N'])
def test_invalid_inputs_stop_before_any_resampling(mutation):
    args = list(inputs())
    if mutation == 'missing_day': args[0] = args[0].iloc[1:]
    if mutation == 'reverse_days': args[1] = args[1].iloc[::-1]
    if mutation == 'naive': args[0].index = args[0].index.tz_localize(None)
    if mutation == 'nonfinite': args[1].iloc[5, 4] = np.nan
    if mutation == 'nonpositive_equity': args[1].iloc[5, 4] = -1.
    if mutation == 'reverse_paths': args[1] = args[1].iloc[:, ::-1]
    if mutation == 'missing_path': args[1] = args[1].iloc[:, :-1]
    if mutation == 'n_gt_N': args[5].loc[0, 'n'] = 31
    with pytest.raises(ScopeError):
        stats.analyze(*args)


@pytest.fixture(scope='module')
def analyzed():
    if stats is None:
        return None
    return stats.analyze(*inputs())


def test_fixed_matrices_hashes_daily_pairing_and_all_seven_gates(analyzed):
    out = analyzed
    s, r, *_ = inputs()
    np.testing.assert_allclose(out['daily'].D.to_numpy(), (s-r.mean(axis=1)).to_numpy(), rtol=1e-14, atol=1e-18)
    assert len(out['daily']) == 1401
    assert len(out['monthly']) == 46
    assert out['metadata']['main']['zero_return_days'] == 201
    assert len(out['tests']) == 2
    assert out['tests'].loc['D', 'mean'] == pytest.approx((s-r.mean(axis=1)).mean())
    assert out['tests'].loc['D', 'p_raw'] == .00001
    assert out['gates'].passed.all()
    assert out['verdict'] == 'is_exploratory_candidate'
    weights = np.random.Generator(np.random.PCG64(2026110902)).choice(
        [-math.sqrt(1.5), -1., -math.sqrt(.5), math.sqrt(.5), 1., math.sqrt(1.5)], size=(99999, 46))
    months = np.random.Generator(np.random.PCG64(2026110903)).integers(0, 46, size=(10000, 46))
    for name, expected in [('webb_weights', weights), ('month_indices', months)]:
        np.testing.assert_array_equal(out['matrices'][name], expected)
        assert out['metadata']['matrices'][name]['sha256'] == hashlib.sha256(expected.tobytes(order='C')).hexdigest()
        assert out['metadata']['matrices'][name]['shape'] == list(expected.shape)
    expected_mcse = r.mean().std(ddof=1)/math.sqrt(200)
    assert out['metadata']['mcse_r'] == pytest.approx(expected_mcse)
    assert out['metadata']['inference_conditioning'] == 'fixed_200_random_schedules'
    assert len(out['random_summary']) == 200


def test_identifiability_capacity_missing_volatility_and_strict_mc_boundary():
    strata = pd.DataFrame({'month': ['2020-03']*3, 'utc_hour': [0, 1, 2],
                           'vol_bin': [0, 0, 5], 'N': [100, 5, 20], 'n': [90, 5, 5]})
    out = stats.identifiability(strata, mean_difference=.02, mcse_r=.01)
    assert out['unsaturated_parent_fraction'] == .95
    assert out['finite_positive_v_parent_fraction'] == .95
    assert out['passed'] is False  # Strict >; equality fails.
    strata.loc[1, 'n'] = 20
    strata.loc[1, 'N'] = 20
    out = stats.identifiability(strata, mean_difference=.03, mcse_r=.01)
    assert out['passed'] is False  # Saturated share now exceeds 10%.
    empty = strata.iloc[:0]
    assert stats.identifiability(empty, mean_difference=1., mcse_r=0.)['passed'] is False


def test_disjoint_ledgers_are_never_forced_into_trade_pairs():
    args = list(inputs())
    args[4] = ledger(month_count=30, per_month=2, bps=3.)
    args[4]['entry_time'] += pd.Timedelta(hours=5)
    args[4]['exit_time'] += pd.Timedelta(hours=5)
    stats.validate_inputs(*args)  # Different cost accounts may have disjoint executions.


def test_no_trades_retains_calendar_paths_and_forces_both_p_values():
    args = list(inputs())
    args[0][:] = 0.
    args[1][:] = 0.
    args[2] = ledger(month_count=0)
    args[3][:] = 0.
    args[4] = ledger(month_count=0)
    args[5]['n'] = 0
    args[6]['trade_count'] = 0
    args[6]['exposure_fraction'] = 0.
    out = stats.analyze(*args)
    assert list(out['tests'].p_raw) == [1., 1.]
    assert list(out['tests'].zero_bootstrap_se_count) == [99999, 99999]
    assert out['intervals'].loc['S_trade_net_bps', 'undefined_resamples'] == 10000
    assert out['intervals'].loc['S_sharpe', 'undefined_resamples'] == 10000
    assert len(out['random_summary']) == 200
    assert out['daily'].eq(0).all().all()
    assert out['verdict'] == 'insufficient_sample_evidence'


def test_all_random_paths_without_trades_force_D_only_and_keep_no_evidence_verdict():
    args = list(inputs())
    args[1][:] = 0.
    args[6]['trade_count'] = 0
    args[6]['exposure_fraction'] = 0.
    out = stats.analyze(*args)
    assert out['tests'].loc['S', 'p_raw'] == .00001
    assert out['tests'].loc['D', 'p_raw'] == 1.
    assert 'all_random_paths_have_no_trades' in out['tests'].loc['D', 'forced_p_one_reasons']
    assert bool(out['gates'].loc['U2', 'passed']) is True
    assert bool(out['gates'].loc['U3', 'passed']) is False
    assert out['verdict'] == 'no_evidence_beyond_random'


def test_empty_S_forces_both_hypotheses_even_when_random_accounts_trade():
    args = list(inputs())
    args[0][:] = 0.
    args[2] = ledger(month_count=0)
    args[3][:] = 0.
    args[4] = ledger(month_count=0)
    out = stats.analyze(*args)
    assert list(out['tests'].p_raw) == [1., 1.]
    assert out['tests'].loc['D', 'zero_bootstrap_se_count'] == 0
    assert out['verdict'] == 'insufficient_sample_evidence'


def test_harsh_cost_only_failure_cannot_become_candidate():
    args = list(inputs())
    args[3] = -args[3]
    args[4]['net_bps'] = -5.
    out = stats.analyze(*args)
    assert out['gates'].index[~out['gates'].passed].tolist() == ['U6']
    assert out['verdict'] == 'cost_sensitive_not_candidate'


def test_U1_needs_actual_entry_months_U4_uses_entry_half_U5_uses_original_ledger():
    args = list(inputs())
    args[2] = ledger(month_count=23, per_month=5)
    args[2]['net_bps'] = -1.
    args[2].loc[:14, 'net_bps'] = 100.
    # Exactly three profitable months; removing these leaves actual losing trades.
    out = stats.analyze(*args)
    assert len(args[2]) == 115
    assert bool(out['gates'].loc['U1', 'passed']) is False
    assert bool(out['gates'].loc['U4', 'passed']) is False
    assert bool(out['gates'].loc['U5', 'passed']) is False
    assert out['metadata']['main']['remaining_trade_mean_bps'] == -1.
    assert out['verdict'] == 'insufficient_sample_evidence'


def test_U7_finite_positive_volatility_counts_parents_not_strata():
    strata = pd.DataFrame({'month': ['2020-03']*3, 'utc_hour': [0, 1, 2],
                           'vol_bin': [0, 1, 5], 'N': [100, 100, 100], 'n': [45, 44, 11]})
    out = stats.identifiability(strata, mean_difference=.1, mcse_r=0.)
    assert out['unsaturated_parent_fraction'] == 1.
    assert out['finite_positive_v_parent_fraction'] == .89
    assert out['passed'] is False


def test_random_return_average_is_not_return_on_averaged_equity(analyzed):
    s, r, *_ = inputs()
    account_equity = 100000*(1+r).cumprod()
    mean_equity_return = account_equity.mean(axis=1).pct_change()
    assert np.max(np.abs(mean_equity_return.iloc[1:].to_numpy()-analyzed['daily'].Rbar.iloc[1:].to_numpy())) > 1e-8
    # The independently maintained accounts, and all their zero days, remain intact.
    assert analyzed['random_summary'].loc['R_000', 'total_return'] == pytest.approx(account_equity.iloc[-1, 0]/100000-1)
    assert analyzed['random_summary'].zero_return_days.eq(201).all()


def test_nearconstant_nonzero_variance_stays_defined_without_epsilon_clamp():
    first = np.array([.01])
    second = np.array([.01+1e-12, .01-1e-12, .01+2e-12])
    data = {'counts': np.array([1, 3]), 'sums': np.array([first.sum(), second.sum()]),
            'squares': np.array([first @ first, second @ second]),
            's_days': [first, second], 'r_sums': np.zeros((2, 2)),
            'trade_counts': np.array([1, 1]), 'trade_sums': np.array([1., 1.])}
    out = stats.month_intervals(data, np.array([[0, 1], [1, 1], [0, 0]]))
    samples = [np.concatenate([first, second]), np.concatenate([second, second])]
    reference = [math.sqrt(365)*x.mean()/x.std(ddof=1) for x in samples]
    assert out.loc['S_sharpe', 'undefined_resamples'] == 1
    assert out.loc['S_sharpe', 'lower'] == pytest.approx(np.quantile(reference, .025, method='linear'))
    assert out.loc['S_sharpe', 'upper'] == pytest.approx(np.quantile(reference, .975, method='linear'))
    assert stats._daily_metrics(samples[0])['annualized_sharpe'] == pytest.approx(reference[0])

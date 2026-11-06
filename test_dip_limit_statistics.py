"""Synthetic mathematical checks; no market files or historical ledgers."""
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

from dip_limit_study import statistics as stats
from dip_limit_study.config import HALF, IS_END, MAIN_START, VERSIONS
from dip_limit_study.scope import ScopeError

DAYS = pd.date_range(MAIN_START, IS_END, freq='D', inclusive='left')
MONTHS = pd.date_range(MAIN_START, IS_END, freq='MS', inclusive='left')


def ledger(bps=10., per_month=3):
    entries = pd.DatetimeIndex([month + pd.Timedelta(days=3+j, hours=1)
                               for month in MONTHS for j in range(per_month)])
    if not len(entries):
        entries = pd.DatetimeIndex([], tz='UTC')
    return pd.DataFrame({'entry_time': entries,
                         'exit_time': entries + pd.Timedelta(hours=2),
                         'net_bps': np.full(len(entries), bps)})


def invented_inputs():
    # Deliberately idle days and unequal month lengths: a full calendar is material.
    phase = np.arange(len(DAYS))
    active = phase % 7 != 0
    main = pd.DataFrame({'M_market': active*(.0001 + np.sin(phase)*.00001),
                         'L_limit': active*(.00016 + np.sin(phase)*.000012)}, index=DAYS)
    trades = {v: ledger() for v in VERSIONS}
    stress = pd.DataFrame({'main': main.L_limit*.9, 'harsh': main.L_limit*.7}, index=DAYS)
    stress_trades = {name: ledger() for name in ('main', 'harsh')}
    events = pd.DataFrame({
        'decision_time': [MONTHS[0]+pd.Timedelta(days=30, hours=23),
                          MONTHS[1]+pd.Timedelta(hours=1),
                          MONTHS[1]+pd.Timedelta(days=3)],
        'M_net_bps': [10., -20., 30.], 'L_net_bps': [14., 0., 0.],
        'difference_bps': [4., 20., -30.], 'L_filled': [True, False, False],
        'L_status': ['filled', 'expired', 'post_only_rejected'],
        'L_wait_minutes': [3., np.nan, np.nan]})
    return main, trades, main*.7, trades, stress, stress_trades, events


@pytest.fixture(scope='module')
def complete_result():
    return stats.analyze(*invented_inputs())


def test_webb_small_matrix_matches_hand_tails_and_zero_se_rule():
    # G=2,N=3,S=(1,3),K=(1,2): mean=4/3,se=2/9,t=6.
    # Supplied draws yield t*=6,-6,0,0; last two have zero cluster SE.
    weights = np.array([[1., 1.], [-1., -1.], [1., 2/3], [-1., -2/3]])
    one = stats.webb_test([1., 3.], [1., 2.], weights, two_sided=False, has_trades=True)
    two = stats.webb_test([1., 3.], [1., 2.], weights, two_sided=True, has_trades=True)
    assert one['mean'] == pytest.approx(4/3)
    assert one['se'] == pytest.approx(2/9)
    assert one['t'] == pytest.approx(6.)
    assert one['p_raw'] == .4
    assert two['p_raw'] == .6
    assert one['zero_bootstrap_se_count'] == 2
    assert stats.webb_test([1., 3.], [1., 2.], weights,
                           two_sided=True, has_trades=False)['p_raw'] == 1.
    assert stats.webb_test([1., 2.], [1., 2.], weights,
                           two_sided=False, has_trades=True)['p_raw'] == 1.


def test_holm_fixed_family_ties_and_monotone_adjustment():
    out = stats.holm_three({'M': .02, 'L': .02, 'D': .04})
    assert [out[k]['rank'] for k in ('M', 'L', 'D')] == [1, 2, 3]
    assert [out[k]['p_holm'] for k in ('M', 'L', 'D')] == [.06, .06, .06]
    assert [out[k]['holm_threshold'] for k in ('M', 'L', 'D')] == [.05/3, .025, .05]
    assert not any(x['reject'] for x in out.values())
    with pytest.raises(ValueError):
        stats.holm_three({'M': .001, 'L': .002})


def test_month_ratios_use_calendar_days_and_trade_counts_not_average_months():
    # Daily samples [1,3] and [9], trades [2] and [10,20].
    data = {'counts': [2, 1], 'sums': [4, 9], 'squares': [10, 81],
            'trade_counts': [1, 2], 'trade_sums': [2, 30]}
    out = stats.month_intervals(data, np.array([[0, 0], [0, 1], [1, 1]]))
    assert out['daily_mean']['estimate'] == pytest.approx(13/3)
    assert out['daily_mean']['lower'] == pytest.approx(2 + .05*(13/3-2))
    assert out['trade_net_bps']['estimate'] == pytest.approx(32/3)
    assert out['trade_net_bps']['upper'] == pytest.approx(15-.05*(15-32/3))
    assert out['annualized_sharpe']['discarded'] == 1
    assert out['annualized_sharpe']['estimate'] == pytest.approx(math.sqrt(365)*(13/3)/math.sqrt(52/3))


def test_entry_month_top_three_ties_remove_earlier_month_and_empty_half_fails():
    entries = pd.to_datetime(['2020-03-03', '2020-04-03', '2020-05-03', '2020-06-03'], utc=True)
    trades = pd.DataFrame({'entry_time': entries, 'exit_time': entries+pd.Timedelta(hours=1),
                           'net_bps': [10., 10., 10., 2.]})
    diag = stats.trade_diagnostics(trades)
    assert diag['removed_best_months'] == ['2020-03', '2020-04', '2020-05']
    assert diag['remaining_trade_mean_bps'] == 2.
    assert diag['Q4'] is True
    assert diag['Q3'] is False
    assert diag['half_trade_mean_bps'] == [8., None]


def test_events_zero_unfilled_and_decision_month_ratio_hand_calculated():
    events = invented_inputs()[-1]
    # March=[4], April=[20,-30]; neither trade exit nor fill month is used.
    ci = stats.event_interval(events, np.array([[0, 0], [0, 1], [1, 1], [2, 2]]))
    assert ci['estimate'] == -2.
    assert ci['lower'] == pytest.approx(-4.85)
    assert ci['upper'] == pytest.approx(3.7)
    assert ci['valid_resamples'] == 3
    assert ci['discarded'] == 1
    diag = stats.event_diagnostics(events)
    assert diag['father_count'] == 3
    assert diag['filled_pair_contribution_bps'] == pytest.approx(4/3)
    assert diag['unfilled_opportunity_contribution_bps'] == pytest.approx(-10/3)
    assert diag['filled_pair_contribution_bps'] + diag['unfilled_opportunity_contribution_bps'] == pytest.approx(-2.)
    assert diag['unfilled_M_mean_net_bps'] == 5.
    assert diag['fill_rate'] == pytest.approx(1/3)


def test_public_analysis_fixed_draws_pairing_and_json(complete_result):
    out = complete_result
    main, _, _, _, _, _, events = invented_inputs()
    weights = np.random.Generator(np.random.PCG64(2026110801)).choice(
        [-math.sqrt(1.5), -1., -math.sqrt(.5), math.sqrt(.5), 1., math.sqrt(1.5)], size=(99999, 46))
    indices = np.random.Generator(np.random.PCG64(2026110802)).integers(0, 46, size=(10000, 46))
    assert out['draw_hashes']['webb'] == hashlib.sha256(weights.tobytes(order='C')).hexdigest()
    assert out['draw_hashes']['month_indices'] == hashlib.sha256(indices.tobytes(order='C')).hexdigest()
    assert out['calendar']['days'] == 1401 and out['calendar']['months'] == 46
    assert out['family'] == ['M', 'L', 'D']
    assert out['tests']['D']['mean'] == pytest.approx((main.L_limit-main.M_market).mean())
    assert out['tests']['D']['p_raw'] == .00001
    assert out['conclusions']['relative']['qualified_winner'] == 'L_limit'
    assert out['metrics']['M_market']['zero_return_days'] > 0
    assert out['gates']['L_limit']['Q7']['passed'] is True
    assert out['intervals']['event_paired_difference']['estimate'] == pytest.approx(events.difference_bps.mean())
    # Independently concatenate each selected month's sufficient statistics.
    labels = main.index.strftime('%Y-%m')
    groups = [main.loc[labels == m.strftime('%Y-%m')] for m in MONTHS]
    day_counts = np.array([len(g) for g in groups])
    differences = np.array([(g.L_limit-g.M_market).sum() for g in groups])
    boot_means = differences[indices].sum(axis=1)/day_counts[indices].sum(axis=1)
    assert out['intervals']['paired_difference']['lower'] == pytest.approx(np.quantile(boot_means, .025, method='linear'))
    json.dumps(out, allow_nan=False)


def test_nontrivial_webb_tests_share_one_registered_matrix():
    args = list(invented_inputs())
    main = args[0].copy()
    labels = main.index.strftime('%Y-%m')
    for i, label in enumerate(MONTHS.strftime('%Y-%m')):
        main.loc[labels == label, 'M_market'] = .00001 + math.sin(i)*.00012
        main.loc[labels == label, 'L_limit'] = .000015 + math.sin(i)*.000072 + math.cos(i)*.00003
    args[0] = main
    out = stats.analyze(*args)
    weights = np.random.Generator(np.random.PCG64(2026110801)).choice(
        [-math.sqrt(1.5), -1., -math.sqrt(.5), math.sqrt(.5), 1., math.sqrt(1.5)], size=(99999, 46))
    for name, series in [('M', main.M_market), ('L', main.L_limit), ('D', main.L_limit-main.M_market)]:
        groups = [series.loc[labels == label] for label in MONTHS.strftime('%Y-%m')]
        counts = np.array([len(g) for g in groups])
        sums = np.array([g.sum() for g in groups])
        mean = sums.sum()/1401
        se = math.sqrt(46/45*np.sum((sums-mean*counts)**2))/1401
        null_sums = weights*sums
        null_mean = null_sums.sum(axis=1)/1401
        null_se = np.sqrt(46/45*np.sum((null_sums-null_mean[:, None]*counts)**2, axis=1))/1401
        null_t = null_mean/null_se
        tail = np.abs(null_t) >= abs(mean/se) if name == 'D' else null_t >= mean/se
        expected = (1+int(tail.sum()))/100000
        assert .00001 < expected < .99
        assert out['tests'][name]['p_raw'] == expected
        assert out['tests'][name]['mcse'] == pytest.approx(math.sqrt(expected*(1-expected)/99999))


@pytest.mark.parametrize('bad_kind', ['missing_day', 'reordered_stress', 'nan_day', 'overlap', 'event_zero', 'event_difference', 'flags'])
def test_invalid_inputs_fail_closed(bad_kind):
    args = list(invented_inputs())
    flags = None
    if bad_kind == 'missing_day':
        args[0] = args[0].iloc[1:]
    elif bad_kind == 'reordered_stress':
        args[4] = args[4].iloc[::-1]
    elif bad_kind == 'nan_day':
        args[2] = args[2].copy()
        args[2].iloc[0, 0] = np.nan
    elif bad_kind == 'overlap':
        args[1]['M_market'] = args[1]['M_market'].copy()
        args[1]['M_market'].loc[0, 'exit_time'] += pd.Timedelta(days=2)
    elif bad_kind == 'event_zero':
        args[6].loc[1, ['L_net_bps', 'difference_bps']] = [1., 21.]
    elif bad_kind == 'event_difference':
        args[6].loc[0, 'difference_bps'] = 99.
    else:
        flags = {'M_market': False, 'L_limit': 1}
    with pytest.raises(ScopeError):
        stats.analyze(*args, failed_versions=flags)


def test_no_trade_zero_returns_keeps_calendar_and_forces_family_p_one():
    args = list(invented_inputs())
    args[0] = args[0]*0
    args[1] = {v: ledger(per_month=0) for v in VERSIONS}
    args[6] = args[6].iloc[:0]
    out = stats.analyze(*args)
    assert all(test['p_raw'] == 1. for test in out['tests'].values())
    assert out['tests']['D']['zero_bootstrap_se_count'] == 99999
    assert out['intervals']['annualized_sharpe']['M_market']['discarded'] == 10000
    assert out['intervals']['trade_net_bps']['L_limit']['discarded'] == 10000
    assert out['intervals']['event_paired_difference']['discarded'] == 10000
    assert out['conclusions']['versions']['M_market'] == 'insufficient_evidence'
    assert out['conclusions']['relative']['stop_current_hypothesis'] is True
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize('failure', ['stress_bankruptcy', 'execution_failure', 'stress_sample', 'stress_harsh_loss', 'stress_no_increment'])
def test_limit_stress_failure_cannot_create_qualified_relative_winner(failure):
    args = list(invented_inputs())
    flags = None
    if failure == 'stress_bankruptcy':
        args[4].loc[DAYS[0], 'harsh'] = -1.
    elif failure == 'execution_failure':
        flags = {'M_market': False, 'L_limit': True}
    elif failure == 'stress_sample':
        args[5]['harsh'] = ledger(per_month=1)
    elif failure == 'stress_harsh_loss':
        args[5]['harsh'] = ledger(bps=-1.)
    else:
        args[4]['main'] = args[0].M_market
    out = stats.analyze(*args, failed_versions=flags)
    assert out['conclusions']['versions']['M_market'] == 'is_exploratory_candidate'
    assert out['conclusions']['versions']['L_limit'] != 'is_exploratory_candidate'
    assert out['conclusions']['relative']['qualified_winner'] is None
    if failure in ('stress_bankruptcy', 'execution_failure'):
        assert out['disqualified_versions']['L_limit'] is True
    else:
        assert out['gates']['L_limit']['Q7']['passed'] is False
        assert out['conclusions']['versions']['L_limit'] == 'fill_sensitive_not_candidate'


def test_q6_uses_same_day_half_directions_and_winners_own_qualification():
    gates = {v: {f'Q{i}': {'passed': True} for i in range(1, 8)} for v in VERSIONS}
    flags = {v: False for v in VERSIONS}
    opposite = stats.relative_conclusion(.01, [.02, -.001], .001, gates, flags)
    assert opposite['Q6'] is False
    assert opposite['qualified_winner'] is None
    negative = stats.relative_conclusion(-.01, [-.02, -.001], .001, gates, flags)
    assert negative['qualified_winner'] == 'M_market'
    gates['M_market']['Q5']['passed'] = False
    assert stats.candidate_status(gates['M_market'], version='M_market', disqualified=False) == 'cost_sensitive_not_candidate'
    failed_own = stats.relative_conclusion(-.01, [-.02, -.001], .001, gates, flags)
    assert failed_own['Q6'] is True
    assert failed_own['qualified_winner'] is None
    gates['M_market']['Q1']['passed'] = False
    assert stats.candidate_status(gates['M_market'], version='M_market', disqualified=False) == 'insufficient_evidence'


@pytest.mark.parametrize('failure', ['entry_months', 'negative_bps', 'negative_half', 'three_month_concentration', 'harsh_loss'])
def test_main_candidate_gates_use_actual_portfolio_entries_and_net_results(failure):
    args = list(invented_inputs())
    # Main and harsh ledgers are deliberately separated for this check.
    args[1] = {v: t.copy() for v, t in args[1].items()}
    args[3] = {v: t.copy() for v, t in args[3].items()}
    expected_gate = {'entry_months': 'Q1', 'negative_bps': 'Q2', 'negative_half': 'Q3',
                     'three_month_concentration': 'Q4', 'harsh_loss': 'Q5'}[failure]
    if failure == 'entry_months':
        args[1]['M_market'] = ledger(per_month=6).iloc[:120]
    elif failure == 'negative_bps':
        args[1]['M_market']['net_bps'] = -1.
    elif failure == 'negative_half':
        second_half = args[1]['M_market'].entry_time >= HALF
        args[1]['M_market'].loc[second_half, 'net_bps'] = -1.
    elif failure == 'three_month_concentration':
        args[1]['M_market']['net_bps'] = -1.
        args[1]['M_market'].loc[0:8, 'net_bps'] = 100.
    else:
        args[3]['M_market']['net_bps'] = -1.
    out = stats.analyze(*args)
    assert out['gates']['M_market'][expected_gate]['passed'] is False
    assert out['conclusions']['versions']['M_market'] != 'is_exploratory_candidate'
    if failure == 'entry_months':
        assert out['gates']['M_market']['Q1']['trade_count'] == 120
        assert out['gates']['M_market']['Q1']['entry_month_count'] == 20
        assert out['conclusions']['versions']['M_market'] == 'insufficient_evidence'
    elif failure == 'harsh_loss':
        assert out['conclusions']['versions']['M_market'] == 'cost_sensitive_not_candidate'


def test_public_q6_halves_use_paired_daily_returns_despite_positive_trade_halves():
    args = list(invented_inputs())
    early = args[0].index < HALF
    args[0]['L_limit'] = args[0].M_market + np.where(early, .0005, -.000001)
    out = stats.analyze(*args)
    assert out['gates']['L_limit']['Q3']['passed'] is True
    assert out['tests']['D']['p_holm'] <= .05
    assert out['conclusions']['relative']['half_mean_paired_differences'] == pytest.approx([.0005, -.000001])
    assert out['gates']['L_limit']['Q6']['passed'] is False
    assert out['conclusions']['relative']['qualified_winner'] is None

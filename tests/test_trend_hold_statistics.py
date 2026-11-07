"""Synthetic checks of the frozen paired inference and candidate rules."""
from hashlib import sha256
import math

import numpy as np
import pandas as pd
import pytest

from trend_hold_study import statistics as stats


def _calendar():
    return pd.date_range('2020-03-01', '2023-12-31', freq='D', tz='UTC')


def _states(calendar):
    states = np.zeros(len(calendar), dtype=bool)
    midpoint = calendar.get_loc(pd.Timestamp('2022-01-01', tz='UTC'))
    states[:60] = True
    states[midpoint:midpoint + 60] = True
    return states


def _raw_arguments():
    calendar = _calendar()
    pattern = np.arange(len(calendar)) % 13
    t = pd.Series(pattern / 131_072, index=calendar)
    p = pd.Series((pattern - 3) / 262_144, index=calendar)
    return [t, p, t.copy(), p.copy(), pd.DataFrame({'g': _states(calendar)}, index=calendar), 15]


def test_webb_one_sided_null_is_recentered_only_by_draw_mean():
    # Unequal clusters. Observed t = 6, null t = [6, -6, -.6, 0].
    sums = np.array([1., 3.])
    counts = np.array([1, 2])
    weights = np.array([[1, 1], [-1, -1], [1, -1], [3, 2]])
    row = stats._webb(sums, counts, weights, has_trades=True)
    assert row['mean'] == pytest.approx(4 / 3)
    assert row['se'] == pytest.approx(2 / 9)
    assert row['t'] == pytest.approx(6)
    assert row['tail_exceedances'] == 1
    assert row['p_raw'] == pytest.approx(2 / 5)
    assert row['zero_bootstrap_se_count'] == 1


def test_webb_zero_observed_se_forces_p_one():
    row = stats._webb([1, 2], [1, 2], [[1, 1], [1, -1]], has_trades=True)
    assert row['se'] == 0
    assert row['t'] is None
    assert row['p_raw'] == 1
    assert row['zero_bootstrap_se_count'] == 1


def test_webb_zero_draw_se_counts_as_zero_t():
    row = stats._webb([-1, -3], [1, 2], [[3, 2], [2, 3]], has_trades=True)
    assert row['t'] == pytest.approx(-6)
    assert row['p_raw'] == 1
    assert row['zero_bootstrap_se_count'] == 1


def test_holm_retains_two_members_breaks_ties_with_t_and_includes_threshold():
    tied = stats._holm({'D': .025, 'T': .025})
    assert tied['T']['rank'] == 1
    assert tied['D']['rank'] == 2
    assert tied['T']['p_holm'] == tied['D']['p_holm'] == .05
    assert tied['T']['reject'] and tied['D']['reject']
    one_forced = stats._holm({'T': .03, 'D': 1.})
    assert one_forced['T']['p_holm'] == .06
    assert not one_forced['T']['reject']
    reversed_rank = stats._holm({'T': .05, 'D': .02})
    assert reversed_rank['D']['rank'] == 1
    assert reversed_rank['T']['p_holm'] == .05


@pytest.mark.parametrize(
    ('first', 'second', 'near_first', 'near_second'),
    [(.0245, .0509, True, True), (.023, .048, False, False)],
)
def test_mc_boundary_uses_raw_p_and_its_ranked_threshold(first, second, near_first, near_second):
    result = stats._holm({'T': first, 'D': second})
    for name, p, near in [('T', first, near_first), ('D', second, near_second)]:
        assert result[name]['mcse'] == pytest.approx(math.sqrt(p * (1 - p) / 99_999))
        assert result[name]['mc_boundary_sensitive'] is near
    assert result['D']['reject'] is (second <= .05)


def test_intervals_resample_paired_whole_months_with_variable_day_counts():
    daily = np.array([[2., 1., 1.], [4., 3., 1.], [9., 8., 1.]])
    draws = np.array([[0, 0], [0, 1], [1, 1]])
    intervals = stats._month_intervals(daily, [2, 1], draws)
    assert intervals.loc['T_daily_mean', 'lower'] == pytest.approx(3.1)
    assert intervals.loc['T_daily_mean', 'upper'] == pytest.approx(8.8)
    assert intervals.loc['P_daily_mean', 'lower'] == pytest.approx(2.1)
    assert intervals.loc['P_daily_mean', 'upper'] == pytest.approx(7.8)
    assert intervals.loc['D_daily_mean', ['estimate', 'lower', 'upper']].tolist() == [1., 1., 1.]
    assert intervals.loc['D_daily_mean', 'discarded'] == 0
    assert intervals.loc['T_sharpe', 'estimate'] == pytest.approx(5 * math.sqrt(365 / 13))
    assert intervals.loc['P_sharpe', 'estimate'] == pytest.approx(4 * math.sqrt(365 / 13))
    assert intervals.loc['T_sharpe', 'valid_resamples'] == 2
    assert intervals.loc['P_sharpe', 'discarded'] == 1


def test_all_undefined_sharpe_does_not_discard_means():
    intervals = stats._month_intervals(
        np.array([[.001, 0., .001]] * 5), [2, 3], np.array([[0, 0], [1, 1]]),
    )
    for name in ('T_sharpe', 'P_sharpe'):
        assert intervals.loc[name, ['estimate', 'lower', 'upper']].isna().all()
        assert intervals.loc[name, 'discarded'] == 2
    assert intervals.loc['D_daily_mean', 'valid_resamples'] == 2


def _gate_case(change=None):
    t, p, harsh_t, harsh_p, features, _ = _raw_arguments()
    t[:] = harsh_t[:] = 1 / 4096
    p[:] = harsh_p[:] = 1 / 8192
    p_t = p_d = .05
    if change == 'V1':
        features.iloc[0, 0] = False
    elif change == 'V1_off_equal':
        features.iloc[:611, 0] = True
    elif change == 'V1_off':
        features.iloc[:612, 0] = True
    elif change == 'V2':
        p_t = .05001
    elif change == 'V2_compounding':
        t.iloc[:2] = [-.75, .8125]
    elif change == 'V3':
        p_d = .05001
    elif change == 'V4':
        p.iloc[:671] = t.iloc[:671]
    elif change == 'V5':
        p[:] = t
        for start in ['2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01']:
            p.loc[start] -= 1 / 1024
        p.loc['2022-01-01'] -= 1 / 2048
        p.loc['2020-10-01'] += 3 / 2048
    elif change == 'V6':
        harsh_p[:] = harsh_t
    elif change == 'V6_compounding':
        harsh_t.iloc[:2] = [-.75, .8125]
    daily = pd.DataFrame({
        'T': t, 'P': p, 'D': t-p,
        'T_harsh': harsh_t, 'P_harsh': harsh_p, 'D_harsh': harsh_t-harsh_p,
    })
    tests = pd.DataFrame({'p_holm': {'T': p_t, 'D': p_d}})
    monthly = daily.groupby(daily.index.strftime('%Y-%m')).agg(
        day_count=('T', 'size'), T_sum=('T', 'sum'), P_sum=('P', 'sum'), D_sum=('D', 'sum'),
    )
    cumulative = {
        'main': {'T': float((1+t).prod()-1)},
        'harsh': {'T': float((1+harsh_t).prod()-1)},
    }
    return stats._gates(daily, features['g'], cumulative, monthly, tests)


@pytest.mark.parametrize(
    ('change', 'failed_gate'),
    [(None, None), ('V1_off_equal', None), ('V1', 'V1'), ('V1_off', 'V1'), ('V2', 'V2'),
     ('V2_compounding', 'V2'), ('V3', 'V3'), ('V4', 'V4'), ('V5', 'V5'),
     ('V6', 'V6'), ('V6_compounding', 'V6')],
)
def test_all_candidate_gates_are_reported_and_jointly_required(change, failed_gate):
    gates = _gate_case(change)
    assert gates.index.tolist() == ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    assert gates.index[~gates['passed']].tolist() == ([failed_gate] if failed_gate else [])
    if failed_gate is None:
        assert stats._classification(gates) == '在本规则下具有净增量的BTC IS探索性候选'
    elif failed_gate == 'V1':
        assert gates.loc['V1', 'details']['status'] == '状态覆盖不足'
    elif failed_gate == 'V6':
        assert stats._classification(gates) == '成本敏感，仍无候选资格'
    else:
        assert stats._classification(gates) != '成本敏感，仍无候选资格'


def test_top_three_difference_months_break_ties_chronologically():
    details = _gate_case('V5').loc['V5', 'details']
    assert details['removed_months'] == ['2020-03', '2020-04', '2020-05']
    assert details['remaining_D_mean'] == 0
    assert details['remaining_T_mean'] > 0


@pytest.mark.parametrize('bad_input', ['date_missing', 'date_reversed', 'zone', 'feature_date', 'bad_g',
                                      'nan', 'infinite', 'zero_equity', 'negative_equity', 'equity_overflow'])
def test_formal_analysis_rejects_invalid_daily_inputs(bad_input):
    args = _raw_arguments()
    if bad_input == 'date_missing':
        args[1] = args[1].iloc[1:]
    elif bad_input == 'date_reversed':
        args[2] = args[2].iloc[::-1]
    elif bad_input == 'zone':
        args[3] = args[3].tz_convert('Etc/GMT')
    elif bad_input == 'feature_date':
        args[4] = args[4].iloc[::-1]
    elif bad_input == 'bad_g':
        args[4]['g'] = args[4]['g'].astype(int)
    elif bad_input == 'nan':
        args[0].iloc[10] = np.nan
    elif bad_input == 'infinite':
        args[1].iloc[11] = np.inf
    elif bad_input == 'zero_equity':
        args[2].iloc[12] = -1.
    elif bad_input == 'negative_equity':
        args[3].iloc[13] = -1.1
    elif bad_input == 'equity_overflow':
        args[3].iloc[14] = 1e307
    with pytest.raises(ValueError):
        stats.analyze(*args)


def test_formal_analysis_draws_frozen_matrices_and_keeps_no_trade_family():
    args = _raw_arguments()
    args[-1] = 0
    result = stats.analyze(*args)
    assert result['tests'].index.tolist() == ['T', 'D']
    assert result['tests']['p_raw'].tolist() == [1., 1.]
    assert result['tests']['p_holm'].tolist() == [1., 1.]
    assert result['monthly']['day_count'].sum() == 1401
    assert len(result['monthly']) == 46
    weights = result['matrices']['webb_weights']
    months = result['matrices']['month_indices']
    points = [-math.sqrt(1.5), -1., -math.sqrt(.5), math.sqrt(.5), 1., math.sqrt(1.5)]
    expected_weights = np.random.Generator(np.random.PCG64(2026111001)).choice(points, size=(99999, 46))
    expected_months = np.random.Generator(np.random.PCG64(2026111002)).integers(0, 46, size=(10000, 46))
    np.testing.assert_array_equal(weights, expected_weights)
    np.testing.assert_array_equal(months, expected_months)
    for name, matrix in result['matrices'].items():
        info = result['metadata']['matrices'][name]
        assert info['shape'] == list(matrix.shape)
        assert info['dtype'] == str(matrix.dtype)
        assert info['sha256'] == sha256(matrix.tobytes(order='C')).hexdigest()


def test_formal_analysis_identical_policy_returns_force_difference_p_one():
    args = _raw_arguments()
    args[1] = args[0].copy()
    result = stats.analyze(*args)
    difference = result['tests'].loc['D']
    assert difference['p_raw'] == 1
    assert 'identical_main_daily_returns' in difference['forced_p_one_reasons']
    assert not result['gates'].loc['V3', 'passed']

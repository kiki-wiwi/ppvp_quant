"""Synthetic-only independent checks of the registered statistical procedure."""
import hashlib
import importlib
import importlib.util
import json
import math
import unittest

import numpy as np
import pandas as pd


SPEC = importlib.util.find_spec('breakout_study.statistics')
stats = importlib.import_module('breakout_study.statistics') if SPEC else None
VERSIONS = ('P_channel', 'V_vah')
CALENDAR = pd.date_range('2020-03-01', '2023-12-31', freq='D', tz='UTC')
MONTHS = pd.date_range('2020-03-01', '2023-12-01', freq='MS', tz='UTC')


def ledger(month_count=46, per_month=3, bps=12.):
    entries = [m + pd.Timedelta(days=j + 1, hours=1)
               for m in MONTHS[:month_count] for j in range(per_month)]
    return pd.DataFrame({'entry_time': pd.DatetimeIndex(entries, tz='UTC'),
                         'exit_time': pd.DatetimeIndex(entries, tz='UTC') + pd.Timedelta(hours=2),
                         'net_bps': np.full(len(entries), bps)})


def returns(p=.0002, v=.0004):
    # Unequal month lengths and genuine within-month variation, with all dates present.
    wave = np.sin(np.arange(1401) * .17) * .00008
    return pd.DataFrame({'P_channel': p + wave, 'V_vah': v + wave * .8}, index=CALENDAR)


class RegisteredStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(stats, 'breakout_study.statistics must implement the approved statistical procedure')

    def test_cluster_standard_error_and_null_webb_t_match_literal_formula(self):
        sums = np.array([.12, -.03, .05])
        counts = np.array([31., 30., 28.])
        weights = np.array([[1., -1., 1.], [-1., 1., -1.], [0., 0., 0.]])
        result = stats.webb_test(sums, counts, weights, two_sided=False, has_trades=True)
        mean = .14 / 89
        se = math.sqrt(1.5 * sum((s - mean*k)**2 for s, k in zip(sums, counts))) / 89
        observed = mean / se
        ts = []
        for row in weights:
            ss = row * sums
            mu = sum(ss)/89
            sd = math.sqrt(1.5 * sum((s-mu*k)**2 for s,k in zip(ss, counts)))/89
            ts.append(mu/sd if sd else 0.)
        self.assertAlmostEqual(result['mean'], mean)
        self.assertAlmostEqual(result['se'], se)
        self.assertAlmostEqual(result['t'], observed)
        self.assertEqual(result['p_raw'], (1 + sum(t >= observed for t in ts))/4)
        self.assertEqual(result['zero_bootstrap_se_count'], 1)

    def test_single_tail_positive_and_two_sided_absolute_are_distinct(self):
        sums = np.array([-.12, .03, -.05])
        counts = np.array([31., 30., 28.])
        weights = np.array([[1., 1., 1.], [-1., -1., -1.], [1., -1., 1.], [-1., 1., -1.]])
        one = stats.webb_test(sums, counts, weights, two_sided=False, has_trades=True)
        two = stats.webb_test(sums, counts, weights, two_sided=True, has_trades=True)
        # Observed t=-1.0865219; null t values are ±1.0865219, ±2.5500831.
        self.assertEqual(one['p_raw'], .8)
        self.assertEqual(two['p_raw'], 1.)
        self.assertEqual(two['tail'], 'two_sided_absolute')

    def test_observed_degeneracy_and_absent_trades_force_p_one(self):
        weights = np.array([[1., 1., 1.], [0., 0., 0.]])
        for sums, traded in [(np.zeros(3), True), (np.array([1., 2., 4.]), False)]:
            out = stats.webb_test(sums, np.ones(3), weights, two_sided=False, has_trades=traded)
            self.assertEqual(out['p_raw'], 1.)
        out = stats.webb_test(np.zeros(3), np.ones(3), weights, two_sided=False, has_trades=True)
        self.assertEqual(out['zero_bootstrap_se_count'], 2)
        self.assertIsNone(out['t'])

    def test_holm_keeps_three_members_stable_ties_and_cumulative_max(self):
        out = stats.holm_three({'P': .01, 'V': .01, 'D': 1.})
        self.assertEqual([out[x]['rank'] for x in ('P', 'V', 'D')], [1, 2, 3])
        self.assertEqual([out[x]['p_holm'] for x in ('P', 'V', 'D')], [.03, .03, 1.])
        self.assertAlmostEqual(out['V']['holm_threshold'], .025)
        degenerate = stats.holm_three({'P': 1., 'V': .02, 'D': 1.})
        self.assertAlmostEqual(degenerate['V']['p_holm'], .06)
        self.assertFalse(degenerate['V']['reject'])

    def test_month_intervals_use_day_denominator_and_concatenated_sharpe(self):
        # Four days in two unequal synthetic months; bootstrap draws reuse all samples.
        month_data = {'counts': np.array([1, 3]), 'sums': np.array([.01, .09]),
                      'squares': np.array([.0001, .0035]),
                      'trade_counts': np.array([1, 2]), 'trade_sums': np.array([10., 60.])}
        draws = np.array([[0, 1], [1, 1], [0, 0]])
        out = stats.month_intervals(month_data, draws)
        samples = [np.array([.01, .01, .03, .05]), np.array([.01, .03, .05]*2), np.array([.01, .01])]
        expected_means = [.025, .03, .01]
        expected_sharpe = [math.sqrt(365)*float(x.mean())/float(x.std(ddof=1)) for x in samples[:2]]
        self.assertAlmostEqual(out['daily_mean']['lower'], float(np.quantile(expected_means, .025, method='linear')))
        self.assertAlmostEqual(out['daily_mean']['upper'], float(np.quantile(expected_means, .975, method='linear')))
        self.assertAlmostEqual(out['annualized_sharpe']['lower'], float(np.quantile(expected_sharpe, .025, method='linear')))
        self.assertEqual(out['annualized_sharpe']['discarded'], 1)
        self.assertAlmostEqual(out['trade_net_bps']['upper'], float(np.quantile([70/3,30,10], .975, method='linear')))

    def test_empty_trade_months_are_retained_and_empty_draws_discarded_without_refill(self):
        data = {'counts': np.array([31, 30]), 'sums': np.array([0., 0.]),
                'squares': np.array([0., 0.]), 'trade_counts': np.array([1, 0]), 'trade_sums': np.array([-5., 0.])}
        out = stats.month_intervals(data, np.array([[0, 1], [1, 1], [0, 0]]))
        self.assertEqual(out['trade_net_bps']['valid_resamples'], 2)
        self.assertEqual(out['trade_net_bps']['discarded'], 1)
        self.assertEqual(out['trade_net_bps']['fraction_below_zero'], 1.)
        self.assertEqual(out['annualized_sharpe']['discarded'], 3)
        self.assertIsNone(out['annualized_sharpe']['lower'])

    def test_best_three_month_ties_remove_earliest_and_empty_half_fails(self):
        t = ledger(month_count=4, per_month=1)
        t['net_bps'] = [10., 10., 10., -1.]
        metric = stats.trade_diagnostics(t)
        self.assertEqual(metric['removed_best_months'], ['2020-03', '2020-04', '2020-05'])
        self.assertEqual(metric['remaining_trade_mean_bps'], -1.)
        self.assertEqual(metric['half_trade_mean_bps'], [7.25, None])
        self.assertFalse(metric['N3'])
        self.assertFalse(metric['N4'])

    def test_cost_sensitive_status_requires_only_n5_failure_and_relative_needs_winner_qualification(self):
        passed = {name: {'passed': True} for name in ('N1','N2','N3','N4','N5')}
        failed_cost = {k: dict(v) for k,v in passed.items()}
        failed_cost['N5']['passed'] = False
        self.assertEqual(stats.candidate_status(failed_cost, bankrupt=False), 'cost_sensitive_candidate')
        failed_cost['N3']['passed'] = False
        self.assertEqual(stats.candidate_status(failed_cost, bankrupt=False), 'not_candidate')
        failed_cost['N1']['passed'] = False
        self.assertEqual(stats.candidate_status(failed_cost, bankrupt=False), 'insufficient_evidence')
        self.assertEqual(stats.candidate_status(passed, bankrupt=True), 'disqualified_bankruptcy')
        gates = {'P_channel': passed, 'V_vah': failed_cost}
        out = stats.relative_conclusion(.001, [.001, .001], .01, gates, {v: False for v in VERSIONS})
        self.assertTrue(out['N6'])
        self.assertEqual(out['status'], 'relative_difference_only')
        self.assertIsNone(out['qualified_winner'])
        out = stats.relative_conclusion(.001, [.001, -.001], .01, {v: passed for v in VERSIONS}, {v: False for v in VERSIONS})
        self.assertFalse(out['N6'])
        self.assertEqual(out['status'], 'indistinguishable')
        out = stats.relative_conclusion(-.001, [-.001, -.001], .01, {v: passed for v in VERSIONS}, {v: False for v in VERSIONS})
        self.assertEqual(out['qualified_winner'], 'P_channel')

    def test_full_calendar_and_invalid_scope_rejected_before_draws(self):
        from breakout_study.scope import ScopeError
        frame = returns()
        trades = {v: ledger() for v in VERSIONS}
        for bad in [frame.iloc[1:], frame.iloc[::-1], frame.set_axis(frame.index.tz_localize(None)), frame.assign(P_channel=np.nan)]:
            with self.assertRaises(ScopeError):
                stats.validate_inputs(bad, trades, frame, trades)
        bad = {v: ledger() for v in VERSIONS}
        bad['P_channel'].loc[0, 'entry_time'] = pd.Timestamp('2023-12-29', tz='UTC')
        with self.assertRaises(ScopeError):
            stats.validate_inputs(frame, bad, frame, trades)
        bad = {v: ledger() for v in VERSIONS}
        bad['P_channel'].loc[0, 'exit_time'] = pd.Timestamp('2024-01-01', tz='UTC')
        with self.assertRaises(ScopeError):
            stats.validate_inputs(frame, bad, frame, trades)
        stats.validate_inputs(frame, trades, frame, trades)

    def test_fixed_full_draws_shared_paired_effect_and_json_has_no_nonfinite_values(self):
        frame = returns()
        frame.iloc[::7] = 0.
        trades = {v: ledger() for v in VERSIONS}
        out = stats.analyze(frame, trades, frame, trades)
        weights = np.random.Generator(np.random.PCG64(2026110701)).choice(
            np.array([-math.sqrt(1.5), -1., -math.sqrt(.5), math.sqrt(.5), 1., math.sqrt(1.5)]), size=(99999,46))
        indices = np.random.Generator(np.random.PCG64(2026110702)).integers(0,46,size=(10000,46))
        self.assertEqual(out['draw_hashes']['webb'], hashlib.sha256(weights.tobytes(order='C')).hexdigest())
        self.assertEqual(out['draw_hashes']['month_indices'], hashlib.sha256(indices.tobytes(order='C')).hexdigest())
        self.assertEqual(out['calendar']['days'], 1401)
        self.assertEqual(out['calendar']['months'], 46)
        self.assertAlmostEqual(out['tests']['D']['mean'], float((frame.V_vah-frame.P_channel).mean()))
        self.assertEqual(out['tests']['D']['p_raw'], .00001)
        self.assertEqual(out['conclusions']['relative']['qualified_winner'], 'V_vah')
        self.assertGreater(out['metrics']['P_channel']['zero_return_days'], 0)
        # Independently aggregate the selected months, including their unequal day counts.
        groups = [frame.loc[frame.index.strftime('%Y-%m') == m.strftime('%Y-%m')] for m in MONTHS]
        lengths = [len(g) for g in groups]
        sums = [float((g.V_vah-g.P_channel).sum()) for g in groups]
        means = [sum(sums[i] for i in row)/sum(lengths[i] for i in row) for row in indices]
        self.assertAlmostEqual(out['intervals']['paired_difference']['lower'], float(np.quantile(means,.025,method='linear')))
        json.dumps(out, allow_nan=False)

    def test_three_nontrivial_pvalues_use_the_identical_registered_weight_matrix(self):
        frame = returns(p=.00001, v=.000015)
        labels = frame.index.strftime('%Y-%m')
        for i, month in enumerate(MONTHS.strftime('%Y-%m')):
            mask = labels == month
            frame.loc[mask, 'P_channel'] += math.sin(i)*.00012
            frame.loc[mask, 'V_vah'] += math.sin(i)*.000072 + math.cos(i)*.00003
        trades = {v: ledger() for v in VERSIONS}
        out = stats.analyze(frame, trades, frame, trades)
        weights = np.random.Generator(np.random.PCG64(2026110701)).choice(
            [-math.sqrt(1.5), -1., -math.sqrt(.5), math.sqrt(.5), 1., math.sqrt(1.5)], size=(99999,46))
        for name, series in [('P', frame.P_channel), ('V', frame.V_vah), ('D', frame.V_vah-frame.P_channel)]:
            groups = [series.loc[labels == month].to_numpy() for month in MONTHS.strftime('%Y-%m')]
            counts = np.array([len(x) for x in groups])
            sums = np.array([sum(x) for x in groups])
            observed_mean = sum(sums)/1401
            observed_se = math.sqrt(46/45*sum((sums-observed_mean*counts)**2))/1401
            null_sums = weights*sums
            null_means = null_sums.sum(axis=1)/1401
            null_se = np.sqrt(46/45*((null_sums-null_means[:,None]*counts)**2).sum(axis=1))/1401
            null_t = null_means/null_se
            tail = np.abs(null_t) >= abs(observed_mean/observed_se) if name == 'D' else null_t >= observed_mean/observed_se
            expected_p = (1+int(tail.sum()))/100000
            self.assertGreater(expected_p, .00001)
            self.assertLess(expected_p, .99)
            self.assertEqual(out['tests'][name]['p_raw'], expected_p)
            self.assertAlmostEqual(out['tests'][name]['mcse'], math.sqrt(expected_p*(1-expected_p)/99999))

    def test_engine_nanosecond_calendar_matches_registered_calendar(self):
        frame = returns()
        frame.index = frame.index.as_unit('ns')
        trades = {v: ledger() for v in VERSIONS}
        stats.validate_inputs(frame, trades, frame, trades)

    def test_execution_failure_disqualifies_despite_positive_daily_returns(self):
        frame = returns()
        trades = {v: ledger() for v in VERSIONS}
        flags = {'P_channel': False, 'V_vah': True}
        out = stats.analyze(frame, trades, frame, trades, failed_versions=flags)
        self.assertTrue((frame > 0).all().all())
        self.assertFalse(out['metrics']['V_vah']['bankrupt'])
        self.assertEqual(out['execution_failed_versions'], flags)
        self.assertEqual(out['conclusions']['versions']['V_vah'], 'disqualified_bankruptcy')
        self.assertIsNone(out['conclusions']['relative']['qualified_winner'])
        self.assertTrue(out['conclusions']['relative']['directional_version_bankrupt'])
        self.assertEqual(out['conclusions']['relative']['status'], 'relative_difference_only')
        json.dumps(out, allow_nan=False)

    def test_execution_failure_flags_require_exact_version_boolean_mapping(self):
        from breakout_study.scope import ScopeError
        frame = returns()
        trades = {v: ledger() for v in VERSIONS}
        for flags in ({}, {'P_channel': False}, {'P_channel': False, 'V_vah': 1},
                      {'P_channel': False, 'V_vah': False, 'extra': False}, [False, False]):
            with self.subTest(flags=flags), self.assertRaises(ScopeError):
                stats.analyze(frame, trades, frame, trades, failed_versions=flags)

    def test_no_trade_pair_is_forced_p_one_and_bankruptcy_disqualifies(self):
        frame = returns()
        frame.iloc[0,0] = -1.
        trades = {'P_channel': ledger(), 'V_vah': ledger(0)}
        out = stats.analyze(frame, trades, frame, trades)
        self.assertEqual(out['tests']['V']['p_raw'], 1.)
        self.assertEqual(out['tests']['D']['p_raw'], 1.)
        self.assertTrue(out['metrics']['P_channel']['bankrupt'])
        self.assertEqual(out['conclusions']['versions']['P_channel'], 'disqualified_bankruptcy')
        self.assertEqual(out['conclusions']['versions']['V_vah'], 'insufficient_evidence')
        self.assertIsNone(out['intervals']['trade_net_bps']['V_vah']['lower'])
        self.assertIsNone(out['conclusions']['relative']['qualified_winner'])
        json.dumps(out, allow_nan=False)


if __name__ == '__main__':
    unittest.main()

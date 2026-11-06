"""Fixed calendar inference for the approved trend/dip execution experiment.

``analyze`` accepts supplied arrays and ledgers; it performs no I/O and exposes
no seed, replicate-count, family, calendar or significance overrides. The three
reused helpers are pure mathematical routines, not the old study's analyzer.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
import platform
import sys

import numpy as np
import pandas as pd

from breakout_study.statistics import _interval, month_intervals, webb_test
from .config import HALF, INITIAL_CAPITAL, IS_END, LAST_DECISION, MAIN_START, VERSIONS
from .scope import ScopeError

WEBB_SEED = 2026110801
MONTH_SEED = 2026110802
WEBB_REPLICATES = 99_999
MONTH_REPLICATES = 10_000
FAMILY = ('M', 'L', 'D')
ALPHA = .05
STRESS_COSTS = ('main', 'harsh')


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _calendar():
    return pd.date_range(MAIN_START, IS_END, freq='D', inclusive='left')


def _months():
    return pd.date_range(MAIN_START, IS_END, freq='MS', inclusive='left').strftime('%Y-%m').tolist()


def _times(frame, column):
    try:
        values = pd.DatetimeIndex(frame[column])
    except (TypeError, ValueError) as exc:
        raise ScopeError(f'{column}: timestamps required') from exc
    if not len(values):
        return pd.DatetimeIndex([], tz='UTC')
    if values.tz is None or values.hasnans:
        raise ScopeError(f'{column}: finite timezone-aware timestamps required')
    return values.tz_convert('UTC')


def _numbers(frame, columns, label):
    try:
        values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ScopeError(f'{label}: numeric values required') from exc
    if not np.isfinite(values).all():
        raise ScopeError(f'{label}: nonfinite values are forbidden')
    return values


def _validate_returns(frame, names, label):
    if not isinstance(frame, pd.DataFrame) or set(frame.columns) != set(names) or len(frame.columns) != len(names):
        raise ScopeError(f'{label}: exactly {names} daily columns required')
    if (not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != 'UTC'
            or not frame.index.as_unit('ns').equals(_calendar().as_unit('ns'))):
        raise ScopeError(f'{label}: exactly 1,401 ascending UTC calendar days in 46 months required')
    _numbers(frame, names, label)


def _validate_ledger(trades, label, version):
    if not isinstance(trades, pd.DataFrame) or not {'entry_time', 'exit_time', 'net_bps'} <= set(trades.columns):
        raise ScopeError(f'{label}: entry_time, exit_time, net_bps required')
    entries, exits = _times(trades, 'entry_time'), _times(trades, 'exit_time')
    if len(entries):
        last_entry = LAST_DECISION + (pd.Timedelta(minutes=59) if version == 'L_limit' else pd.Timedelta(0))
        if not entries.is_monotonic_increasing or entries.has_duplicates:
            raise ScopeError(f'{label}: entries must be unique and ascending')
        if ((entries < MAIN_START).any() or (entries > last_entry).any()
                or (exits >= IS_END).any() or (exits < entries).any()):
            raise ScopeError(f'{label}: invalid trade scope or time order')
        if (entries[1:] < exits[:-1]).any():
            raise ScopeError(f'{label}: overlapping portfolio positions are forbidden')
        if 'decision_time' in trades.columns:
            decisions = _times(trades, 'decision_time')
            if ((decisions < MAIN_START+pd.Timedelta(hours=1)).any()
                    or (decisions > LAST_DECISION).any() or (decisions > entries).any()):
                raise ScopeError(f'{label}: invalid father decision scope or time order')
    _numbers(trades, ('net_bps',), label)


def _validate_events(events):
    required = {'decision_time', 'M_net_bps', 'L_net_bps', 'difference_bps',
                'L_filled', 'L_status', 'L_wait_minutes'}
    if not isinstance(events, pd.DataFrame) or not required <= set(events.columns):
        raise ScopeError('events: complete paired father-event columns required')
    decisions = _times(events, 'decision_time')
    if len(decisions) and (not decisions.is_monotonic_increasing or decisions.has_duplicates
                          or (decisions < MAIN_START+pd.Timedelta(hours=1)).any()
                          or (decisions > LAST_DECISION).any()):
        raise ScopeError('events: unique ascending in-scope father decision times required')
    values = _numbers(events, ('M_net_bps', 'L_net_bps', 'difference_bps'), 'events')
    if any(not isinstance(v, (bool, np.bool_)) for v in events['L_filled']):
        raise ScopeError('events: L_filled must contain booleans')
    filled = events['L_filled'].to_numpy(dtype=bool)
    if np.any(values[~filled, 1] != 0):
        raise ScopeError('events: every unfilled L outcome must remain zero')
    if not np.allclose(values[:, 2], values[:, 1]-values[:, 0], rtol=1e-12, atol=1e-12):
        raise ScopeError('events: difference_bps must equal L_net_bps minus M_net_bps')
    try:
        waits = events['L_wait_minutes'].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ScopeError('events: numeric or undefined wait minutes required') from exc
    if np.any(~np.isfinite(waits[filled])) or np.any(waits[filled] < 0) or np.any(waits[filled] >= 60):
        raise ScopeError('events: filled L wait must lie within its 60-minute lifetime')


def validate_inputs(main_returns, main_trades, harsh_returns, harsh_trades,
                    stress_returns, stress_trades, events, *, failed_versions=None):
    """Reject truncation, mismatched calendars, invalid ledgers and altered pairing."""
    if failed_versions is not None and (
            not isinstance(failed_versions, Mapping) or set(failed_versions) != set(VERSIONS)
            or any(not isinstance(v, bool) for v in failed_versions.values())):
        raise ScopeError('failed_versions must map exactly both registered versions to booleans')
    for label, returns, ledgers, names in (
            ('main', main_returns, main_trades, VERSIONS),
            ('harsh', harsh_returns, harsh_trades, VERSIONS),
            ('L_stress', stress_returns, stress_trades, STRESS_COSTS)):
        _validate_returns(returns, names, label)
        if not isinstance(ledgers, Mapping) or set(ledgers) != set(names):
            raise ScopeError(f'{label}: all registered ledgers required, including empty ledgers')
        for name in names:
            _validate_ledger(ledgers[name], f'{label}/{name}', 'L_limit' if label == 'L_stress' else name)
    _validate_events(events)


def holm_three(raw_p):
    """Fixed M/L/D family; exact p ties are broken in that registered order."""
    if set(raw_p) != set(FAMILY) or any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for p in raw_p.values()):
        raise ValueError('Holm requires exactly three finite M/L/D probabilities')
    order = sorted(FAMILY, key=lambda name: (raw_p[name], FAMILY.index(name)))
    result, running = {}, 0.
    for index, name in enumerate(order):
        multiplier = 3-index
        running = min(1., max(running, multiplier*float(raw_p[name])))
        result[name] = {'rank': index+1, 'p_holm': running,
                        'holm_threshold': ALPHA/multiplier, 'reject': running <= ALPHA}
    return result


def trade_diagnostics(trades):
    """Entry-time halves and months, without re-execution or recomputed equity."""
    entries, exits = _times(trades, 'entry_time'), _times(trades, 'exit_time')
    values = trades['net_bps'].to_numpy(dtype=float)
    months = entries.strftime('%Y-%m')
    monthly = pd.Series(values, index=months).groupby(level=0).sum()
    removed = sorted(monthly.index, key=lambda month: (-float(monthly.loc[month]), month))[:3]
    remaining = values[~np.isin(months, removed)]
    halves = [values[entries < HALF], values[entries >= HALF]]
    half_means = [float(part.mean()) if len(part) else None for part in halves]
    remainder = float(remaining.mean()) if len(remaining) else None
    crossed = months != exits.strftime('%Y-%m')
    return {'trade_count': len(values), 'entry_month_count': len(monthly),
            'mean_net_bps': float(values.mean()) if len(values) else None,
            'half_trade_counts': [len(part) for part in halves], 'half_trade_mean_bps': half_means,
            'entry_month_net_bps_sums': {str(k): float(v) for k, v in monthly.items()},
            'removed_best_months': removed, 'remaining_trade_count': len(remaining),
            'remaining_trade_mean_bps': remainder,
            'cross_month_trade_count': int(np.count_nonzero(crossed)),
            'cross_month_trade_fraction': float(np.mean(crossed)) if len(values) else None,
            'Q3': all(v is not None and v > 0 for v in half_means),
            'Q4': remainder is not None and remainder > 0}


def _daily_metrics(series):
    values = series.to_numpy(dtype=float)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        equity = INITIAL_CAPITAL*np.cumprod(1+values)
    nonpositive = (values <= -1) | (equity <= 0)
    bankrupt = bool(nonpositive.any())
    numeric_failure = bool((~np.isfinite(equity)).any())
    terminal = None if bankrupt or numeric_failure else float(equity[-1])
    sd = float(np.std(values, ddof=1))
    return {'daily_mean': float(values.mean()),
            'annualized_sharpe': math.sqrt(365)*float(values.mean())/sd if sd > 0 else None,
            'total_return': terminal/INITIAL_CAPITAL-1 if terminal is not None else None,
            'final_equity': terminal, 'bankrupt': bankrupt, 'equity_numerically_undefined': numeric_failure,
            'first_nonpositive_equity_day': str(series.index[np.flatnonzero(nonpositive)[0]].date()) if bankrupt else None,
            'zero_return_days': int(np.count_nonzero(values == 0))}


def _month_data(series, trades):
    months = _months()
    labels = series.index.strftime('%Y-%m')
    grouped = series.groupby(labels)
    entry_months = _times(trades, 'entry_time').strftime('%Y-%m')
    trade_values = pd.Series(trades['net_bps'].to_numpy(dtype=float), index=entry_months)
    trade_grouped = trade_values.groupby(level=0)
    return {'counts': grouped.size().reindex(months, fill_value=0).to_numpy(dtype=float),
            'sums': grouped.sum().reindex(months, fill_value=0).to_numpy(dtype=float),
            'squares': (series*series).groupby(labels).sum().reindex(months, fill_value=0).to_numpy(dtype=float),
            'trade_counts': trade_grouped.size().reindex(months, fill_value=0).to_numpy(dtype=float),
            'trade_sums': trade_grouped.sum().reindex(months, fill_value=0).to_numpy(dtype=float)}


def event_interval(events, draws):
    """Paired bps averaged over all sampled fathers, including every unfilled L."""
    labels = _times(events, 'decision_time').strftime('%Y-%m')
    values = pd.Series(events['difference_bps'].to_numpy(dtype=float), index=labels)
    grouped = values.groupby(level=0)
    counts = grouped.size().reindex(_months(), fill_value=0).to_numpy(dtype=float)
    sums = grouped.sum().reindex(_months(), fill_value=0).to_numpy(dtype=float)
    draws = np.asarray(draws, dtype=np.int64)
    denominator = counts[draws].sum(axis=1)
    numerator = sums[draws].sum(axis=1)
    estimates = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return _interval(estimates, estimate=float(values.mean()) if len(values) else None)


def event_diagnostics(events):
    filled = events['L_filled'].to_numpy(dtype=bool)
    m = events['M_net_bps'].to_numpy(dtype=float)
    l = events['L_net_bps'].to_numpy(dtype=float)
    differences = events['difference_bps'].to_numpy(dtype=float)
    count = len(events)
    def mean(values):
        return float(np.mean(values)) if len(values) else None
    return {'father_count': count, 'filled_count': int(filled.sum()),
            'unfilled_count': int((~filled).sum()), 'fill_rate': float(filled.mean()) if count else None,
            'mean_M_net_bps': mean(m), 'mean_L_net_bps': mean(l),
            'mean_difference_bps': mean(differences),
            'filled_pair_contribution_bps': float(differences[filled].sum()/count) if count else None,
            'unfilled_opportunity_contribution_bps': float((-m[~filled]).sum()/count) if count else None,
            'filled_M_mean_net_bps': mean(m[filled]), 'unfilled_M_mean_net_bps': mean(m[~filled]),
            'filled_pair_mean_difference_bps': mean(differences[filled]),
            'mean_filled_wait_minutes': mean(events.loc[filled, 'L_wait_minutes'].to_numpy(dtype=float)),
            'L_status_counts': {str(k): int(v) for k, v in events['L_status'].value_counts(dropna=False).items()},
            'contribution_denominator': 'all_eligible_fathers',
            'independent_overlapping_events_are_not_a_tradable_portfolio': True}


def _required_gates(version):
    return ('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q7') if version == 'L_limit' else ('Q1', 'Q2', 'Q3', 'Q4', 'Q5')


def _passed(gates, version):
    return all(gates[name]['passed'] for name in _required_gates(version))


def candidate_status(gates, *, version, disqualified):
    if disqualified:
        return 'disqualified_execution_or_equity_failure'
    if not gates['Q1']['passed']:
        return 'insufficient_evidence'
    failures = [name for name in _required_gates(version) if not gates[name]['passed']]
    if not failures:
        return 'is_exploratory_candidate'
    if failures == ['Q5']:
        return 'cost_sensitive_not_candidate'
    if failures == ['Q7']:
        return 'fill_sensitive_not_candidate'
    return 'not_candidate'


def relative_conclusion(mean_difference, half_differences, adjusted_p, gates, disqualified):
    direction = 1 if mean_difference > 0 else -1 if mean_difference < 0 else 0
    directional_version = 'L_limit' if direction > 0 else 'M_market' if direction < 0 else None
    consistent = direction != 0 and all(v is not None and v*direction > 0 for v in half_differences)
    significant = adjusted_p <= ALPHA
    q6 = bool(significant and consistent)
    qualified = {v: bool(_passed(gates[v], v) and not disqualified[v]) for v in VERSIONS}
    winner = directional_version if q6 and qualified.get(directional_version, False) else None
    if winner:
        status = 'qualified_policy_increment_supported'
    elif all(qualified.values()):
        status = 'indistinguishable'
    elif not any(qualified.values()):
        status = 'stop_current_hypothesis'
    elif significant:
        status = 'relative_difference_only'
    else:
        status = 'no_supported_relative_increment'
    return {'status': status, 'Q6': q6, 'statistically_significant': significant,
            'directional_version': directional_version, 'qualified_winner': winner,
            'qualified_versions': qualified,
            'mean_paired_difference': _finite(mean_difference),
            'half_mean_paired_differences': [_finite(v) for v in half_differences],
            'half_directions_agree': bool(consistent),
            'directional_version_failed_gates': [q for q in _required_gates(directional_version)
                                                 if not gates[directional_version][q]['passed']] if directional_version else [],
            'directional_version_disqualified': bool(disqualified[directional_version]) if directional_version else False,
            'limit_relative_improvement_supported': winner == 'L_limit',
            'market_preferred_without_passive_evidence': winner == 'M_market',
            'stop_current_hypothesis': not any(qualified.values())}


def _positive_metrics(metrics):
    return bool(metrics['total_return'] is not None and metrics['total_return'] > 0
                and metrics['mean_net_bps'] is not None and metrics['mean_net_bps'] > 0)


def _sample_gate(metrics):
    return {'passed': metrics['trade_count'] >= 100 and metrics['entry_month_count'] >= 24,
            'trade_count': metrics['trade_count'], 'entry_month_count': metrics['entry_month_count'],
            'minimum_trades': 100, 'minimum_entry_months': 24}


def analyze(main_returns, main_trades, harsh_returns, harsh_trades,
            stress_returns, stress_trades, events, *, failed_versions: Mapping[str, bool] | None = None):
    """Run the fixed M/L/D family and descriptive intervals on supplied inputs.

    ``failed_versions`` is the runtime OR of execution failures across every
    respective cost and stress scenario, including intraminute nonpositive
    equity invisible in daily closes. Stress daily failures also exclude L here.
    """
    validate_inputs(main_returns, main_trades, harsh_returns, harsh_trades,
                    stress_returns, stress_trades, events, failed_versions=failed_versions)
    execution_failed = {v: failed_versions[v] if failed_versions is not None else False for v in VERSIONS}
    webb_values = np.array([-math.sqrt(1.5), -1., -math.sqrt(.5), math.sqrt(.5), 1., math.sqrt(1.5)])
    weights = np.random.Generator(np.random.PCG64(WEBB_SEED)).choice(webb_values, size=(WEBB_REPLICATES, 46))
    draws = np.random.Generator(np.random.PCG64(MONTH_SEED)).integers(0, 46, size=(MONTH_REPLICATES, 46))
    monthly = {v: _month_data(main_returns[v], main_trades[v]) for v in VERSIONS}
    paired = main_returns['L_limit']-main_returns['M_market']
    difference = _month_data(paired, main_trades['M_market'].iloc[:0])
    tests = {}
    for name, version in [('M', 'M_market'), ('L', 'L_limit'), ('D', None)]:
        data = difference if name == 'D' else monthly[version]
        has_trades = all(len(main_trades[v]) > 0 for v in VERSIONS) if name == 'D' else len(main_trades[version]) > 0
        tests[name] = webb_test(data['sums'], data['counts'], weights, two_sided=name == 'D', has_trades=has_trades)
        tests[name]['hypothesis'] = 'mean(L_limit-M_market)=0' if name == 'D' else f'mean({version})<=0'
    holm = holm_three({name: tests[name]['p_raw'] for name in FAMILY})
    for name in FAMILY:
        tests[name].update(holm[name])
        tests[name]['mc_boundary_sensitive'] = abs(tests[name]['p_raw']-tests[name]['holm_threshold']) < 2*tests[name]['mcse']

    intervals = {'daily_mean': {}, 'annualized_sharpe': {}, 'trade_net_bps': {},
                 'paired_difference': month_intervals(difference, draws)['daily_mean'],
                 'event_paired_difference': event_interval(events, draws)}
    metrics, harsh_metrics, gates = {}, {}, {}
    stress_metrics = {cost: {**_daily_metrics(stress_returns[cost]), **trade_diagnostics(stress_trades[cost])}
                      for cost in STRESS_COSTS}
    disqualified, failure_reasons = {}, {}
    for name, version in zip(('M', 'L'), VERSIONS):
        ci = month_intervals(monthly[version], draws)
        for metric in ('daily_mean', 'annualized_sharpe', 'trade_net_bps'):
            intervals[metric][version] = ci[metric]
        m = metrics[version] = {**_daily_metrics(main_returns[version]), **trade_diagnostics(main_trades[version])}
        h = harsh_metrics[version] = {**_daily_metrics(harsh_returns[version]), **trade_diagnostics(harsh_trades[version])}
        scenarios = [('main', m), ('harsh', h)]
        if version == 'L_limit':
            scenarios += [(f'stress_{cost}', stress_metrics[cost]) for cost in STRESS_COSTS]
        failure_reasons[version] = [f'{label}:nonpositive_equity' for label, value in scenarios if value['bankrupt']]
        failure_reasons[version] += [f'{label}:undefined_equity' for label, value in scenarios if value['equity_numerically_undefined']]
        if execution_failed[version]:
            failure_reasons[version].append('runtime_any_cost_or_stress_execution_failure')
        disqualified[version] = bool(failure_reasons[version])
        gates[version] = {
            'Q1': _sample_gate(m),
            'Q2': {'passed': bool(tests[name]['reject'] and _positive_metrics(m)),
                   'p_holm': tests[name]['p_holm'], 'total_return': m['total_return'], 'mean_net_bps': m['mean_net_bps']},
            'Q3': {'passed': m['Q3'], 'half_trade_mean_bps': m['half_trade_mean_bps'], 'half_trade_counts': m['half_trade_counts']},
            'Q4': {'passed': m['Q4'], 'removed_best_months': m['removed_best_months'],
                   'remaining_trade_mean_bps': m['remaining_trade_mean_bps'], 'remaining_trade_count': m['remaining_trade_count']},
            'Q5': {'passed': _positive_metrics(h), 'harsh_total_return': h['total_return'], 'harsh_mean_net_bps': h['mean_net_bps']}}
    stress_arms = {cost: {**_sample_gate(stress_metrics[cost]),
                         'sample_passed': _sample_gate(stress_metrics[cost])['passed'],
                         'positive_performance': _positive_metrics(stress_metrics[cost]),
                         'total_return': stress_metrics[cost]['total_return'],
                         'mean_net_bps': stress_metrics[cost]['mean_net_bps']}
                   for cost in STRESS_COSTS}
    for arm in stress_arms.values():
        arm['passed'] = bool(arm['sample_passed'] and arm['positive_performance'])
    stress_increment = float((stress_returns['main']-main_returns['M_market']).mean())
    gates['L_limit']['Q7'] = {'passed': bool(all(arm['passed'] for arm in stress_arms.values()) and stress_increment > 0),
                             'arms': stress_arms, 'stress_main_minus_M_main_daily_mean': stress_increment,
                             'positive_stress_increment': stress_increment > 0, 'additional_significance_test': False}
    gates['M_market']['Q7'] = {'passed': None, 'applicable': False}
    half_differences = [float(paired.loc[paired.index < HALF].mean()), float(paired.loc[paired.index >= HALF].mean())]
    relative = relative_conclusion(float(paired.mean()), half_differences, tests['D']['p_holm'], gates, disqualified)
    for version in VERSIONS:
        gates[version]['Q6'] = {'passed': bool(relative['Q6'] and relative['directional_version'] == version),
                                'p_holm': tests['D']['p_holm'], 'half_mean_paired_differences': half_differences,
                                'directional_version': relative['directional_version'],
                                'supports_qualified_increment': relative['qualified_winner'] == version}
    disagreements = []
    for name, interval in [('M', intervals['daily_mean']['M_market']),
                           ('L', intervals['daily_mean']['L_limit']), ('D', intervals['paired_difference'])]:
        supports = interval['lower'] is not None and (interval['lower'] > 0 or (name == 'D' and interval['upper'] < 0))
        disagreements.append({'hypothesis': name, 'interval_supports_test_direction': bool(supports),
                              'holm_rejects': tests[name]['reject'], 'disagrees': bool(supports != tests[name]['reject'])})
    gate_disagreements = []
    for version in VERSIONS:
        ci_positive = intervals['daily_mean'][version]['lower'] is not None and intervals['daily_mean'][version]['lower'] > 0
        gate_disagreements.append({'version': version, 'daily_mean_interval_positive': bool(ci_positive),
                                   'Q2_passed': gates[version]['Q2']['passed'],
                                   'qualified': relative['qualified_versions'][version],
                                   'interval_vs_Q2_disagrees': bool(ci_positive != gates[version]['Q2']['passed']),
                                   'interval_vs_candidate_disagrees': bool(ci_positive != relative['qualified_versions'][version])})
    return {
        'schema_version': 1,
        'calendar': {'start': MAIN_START.isoformat(), 'end_exclusive': IS_END.isoformat(),
                     'days': 1401, 'months': 46, 'month_order': _months()},
        'estimand': 'full_calendar_portfolio_mean_daily_return', 'family': list(FAMILY),
        'family_size': 3, 'family_alpha': ALPHA, 'tests': tests, 'intervals': intervals,
        'metrics': metrics, 'harsh_metrics': harsh_metrics, 'stress_metrics': stress_metrics,
        'events': event_diagnostics(events), 'gates': gates,
        'execution_failed_versions': execution_failed, 'disqualified_versions': disqualified,
        'failure_reasons': failure_reasons,
        'conclusions': {'versions': {v: candidate_status(gates[v], version=v, disqualified=disqualified[v]) for v in VERSIONS},
                        'relative': relative,
                        'candidate_labels': {'M_market': '市价急跌规则的 IS 探索性候选',
                                             'L_limit': '在登记成交假设下的限价 IS 探索性候选'},
                        'cost_sensitive_label': '成本敏感、不具备资格',
                        'fill_sensitive_label': '成交假设敏感、不具备限价候选资格',
                        'no_sharpe_winner_selection': True},
        'seeds': {'webb': WEBB_SEED, 'month_indices': MONTH_SEED, 'generator': 'NumPy Generator(PCG64)'},
        'draw_hashes': {'webb': hashlib.sha256(weights.tobytes(order='C')).hexdigest(),
                        'month_indices': hashlib.sha256(draws.tobytes(order='C')).hexdigest()},
        'draw_metadata': {'webb_shape': list(weights.shape), 'webb_dtype': str(weights.dtype),
                          'month_indices_shape': list(draws.shape), 'month_indices_dtype': str(draws.dtype),
                          'byte_order': sys.byteorder, 'hash_serialization': 'ndarray.tobytes(order=C)'},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'platform': platform.platform()},
        'disclosures': {
            'interval_p_disagreements': disagreements, 'interval_gate_disagreements': gate_disagreements,
            'mc_boundary_sensitive_hypotheses': [name for name in FAMILY if tests[name]['mc_boundary_sensitive']],
            'mcse_formula': 'sqrt(p_raw*(1-p_raw)/99999)', 'mc_boundary_rule': 'abs(p_raw-step_threshold)<2*MCSE',
            'no_resampling_refill': True, 'intervals_are_not_significance_gates': True,
            'trade_clusters': 'entry_month', 'event_clusters': 'father_decision_month',
            'unfilled_event_L_returns_are_zero': True,
            'same_month_draws_for_M_L_D_trade_and_event_intervals': True,
            'cross_month_trade_fraction': {v: metrics[v]['cross_month_trade_fraction'] for v in VERSIONS},
            'stress_cross_month_trade_fraction': {cost: stress_metrics[cost]['cross_month_trade_fraction'] for cost in STRESS_COSTS},
            'cluster_limitation': 'Month clustering permits within-month dependence; cross-month holdings and market dependence remain.',
            'relative_effect_scope': 'Execution-policy total effect includes prices, waiting, nonfills, fees and portfolio state; it does not isolate maker fees or signal alpha.',
            'stress_limitation': 'Joint stress changes the fill set and is not a performance lower bound or proof of real maker execution.',
            'multiple_research_limitation': 'The three-test Holm family controls this registered family only, not prior research on these already-viewed IS data.',
            'sample_minimum_is_not_power_guarantee': True, 'independent_validation_required': True,
            'event_portfolio_sharpe_prohibited': True,
            'failure_flags_contract': 'Runtime must OR every cost and stress execution failure per version, including intraminute nonpositive equity.',
            'reused_pure_helpers': ['breakout_study.statistics.webb_test',
                                    'breakout_study.statistics.month_intervals', 'breakout_study.statistics._interval']}}

"""Frozen CP3 §8.3 matched random-entry controls.

Pre-unlock implementation choices (no statistical parameters are configurable):
* A valid sigma7d is finite, matching the frozen signal implementation.
* Balance compares matched originals with their 100 draws per trade. It reports
  count, mean, population SD, min/5/25/50/75/95/max, and 24 hour counts.
* If no trade has candidates, the auxiliary randomization p is 1, its observed
  mean is undefined and its null-mean array is empty; G3's empty-sample handling
  belongs to cp3_statistics. Missing trades consume no random numbers.
* Pool lookup uses stable sorting by integer keys; every returned pool retains
  ascending minute-index order. The one RNG per procedure is consumed exactly
  in frozen row-major order; simulation does not consume random numbers.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np
import pandas as pd
from . import engine as E, versions as VS
from .features import Context
from .scope import IS_START, IS_END, ScopeError

K = 100
NULL_DRAWS = 5000
RELAXATION_CLASSES = (
    'month_hour_atr', 'month_6hour_atr', 'month_atr',
    'month_ge20', 'month_1to19', 'missing_month',
)


@dataclass
class MatchDesign:
    records: pd.DataFrame
    sample_indices: np.ndarray
    candidate_indices: np.ndarray
    q1: float
    q2: float
    relaxation_counts: dict


@dataclass
class MatchedControlsResult:
    records: pd.DataFrame
    sample_indices: np.ndarray
    random_net_bps: dict
    relaxation_counts: dict
    balance: dict
    randomization: dict
    quantiles: dict
    candidate_count: int


def _check_scope(ctx: Context, trades: pd.DataFrame):
    if ctx.index.tz is None or (len(ctx.index) and (ctx.index.min() < IS_START or ctx.index.max() >= IS_END)):
        raise ScopeError('matched controls require a UTC BTC IS context')
    w = ctx.index[ctx.window]
    if len(w) and (w.min() < VS.MAIN_START or w.max() > VS.LAST_TRIGGER):
        raise ScopeError('matched controls are limited to the main trigger window')
    for column, low, high in (
        ('trigger_time', VS.MAIN_START, VS.LAST_TRIGGER),
        ('entry_time', VS.MAIN_START, VS.LAST_TRIGGER + pd.Timedelta(minutes=1)),
        ('exit_time', VS.MAIN_START, IS_END),
    ):
        times = pd.DatetimeIndex(trades[column])
        if times.tz is None:
            raise ScopeError(f'{column} must be timezone-aware')
        if len(times) and (times.isna().any() or times.min() < low or times.max() > high):
            raise ScopeError(f'{column} leaves the frozen statistics window')


def _sort_trades(ctx, trades):
    _check_scope(ctx, trades)
    tr = trades.sort_values('trigger_idx', kind='stable').reset_index(drop=True).copy()
    indices = tr['trigger_idx'].to_numpy()
    if len(indices) and (not np.issubdtype(indices.dtype, np.integer) or
                         indices.min() < 0 or indices.max() >= len(ctx.index)):
        raise ValueError('actual trigger indices must be valid integers')
    indices = indices.astype(np.int64)
    if len(indices) and (np.any(np.diff(indices) <= 0) or not ctx.window[indices].all()):
        raise ValueError('actual trades require unique main-window triggers')
    if not pd.DatetimeIndex(tr['trigger_time']).equals(ctx.index[indices]):
        raise ValueError('actual trigger timestamps disagree with minute indices')
    return tr


def _group_pools(indices, keys):
    """Stable grouping preserves ascending indices within each distinct stratum."""
    if not len(indices):
        return {}
    order = np.argsort(keys, kind='stable')
    key = keys[order]
    grouped = indices[order]
    starts = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]
    ends = np.r_[starts[1:], len(indices)]
    return {int(key[a]): grouped[a:b] for a, b in zip(starts, ends)}


def build_match_design(ctx: Context, spec: VS.VersionSpec, main_trades: pd.DataFrame) -> MatchDesign:
    """Draw K=100 matches per actual trade, with exactly the §8.3 relaxation."""
    tr = _sort_trades(ctx, main_trades)
    v = list(VS.VERSIONS).index(spec.name)
    if VS.VERSIONS[spec.name] != spec:
        raise ValueError('matching requires a frozen version specification')
    atr = np.asarray(ctx.atr['atr_at'], float)
    with np.errstate(invalid='ignore', divide='ignore'):
        a = atr / ctx.C
    quantile_mask = ctx.window & ~ctx.bad & np.isfinite(atr)
    if quantile_mask.any():
        if not np.isfinite(a[quantile_mask]).all():
            raise ValueError('ATR quantile population has nonfinite ATR / close')
        q1, q2 = np.quantile(a[quantile_mask], [1 / 3, 2 / 3])
    else:
        q1, q2 = np.nan, np.nan
    band = np.searchsorted(np.array([q1, q2]), a, side='left')
    n = len(ctx.C)
    next_good = np.zeros(n, dtype=bool)
    next_good[:-1] = ~ctx.bad[1:]
    condition = ctx.cond[spec.trigger_rule]
    any_trigger = condition['up1'] | condition['up2'] | condition['dn1'] | condition['dn2']
    eligible = (ctx.window & ~ctx.frame['contaminated'].to_numpy(dtype=bool)
                & np.isfinite(atr) & (atr > 0) & next_good
                & np.isfinite(ctx.base_med) & ~any_trigger)
    if spec.box:
        eligible = eligible & ctx.box_ok
    if spec.trigger_rule == 'z':
        eligible = eligible & np.isfinite(ctx.frame['sigma7d'].to_numpy())
    candidates = np.flatnonzero(eligible)
    month = np.asarray(ctx.index.year * 12 + ctx.index.month, dtype=np.int64)
    hour = np.asarray(ctx.index.hour, dtype=np.int64)
    key_arrays = ((month * 24 + hour) * 3 + band,
                  (month * 4 + hour // 6) * 3 + band,
                  month * 3 + band, month)
    pools = [_group_pools(candidates, key[candidates]) for key in key_arrays]
    rng = np.random.default_rng(VS.SEED_BASE + 100 + v)
    sample_indices = np.full((len(tr), K), -1, dtype=np.int64)
    counts = dict.fromkeys(RELAXATION_CLASSES, 0)
    labels, sizes = [], []
    empty = np.empty(0, dtype=np.int64)
    for row, trigger in enumerate(tr.trigger_idx.to_numpy(dtype=np.int64)):
        pool = empty
        selected = 'missing_month'
        for level in range(4):
            pool = pools[level].get(int(key_arrays[level][trigger]), empty)
            if len(pool) >= 20:
                selected = RELAXATION_CLASSES[level]
                break
            if level == 3:
                selected = 'month_1to19' if len(pool) else 'missing_month'
        labels.append(selected)
        sizes.append(len(pool))
        counts[selected] += 1
        if len(pool):
            sample_indices[row] = pool[rng.integers(0, len(pool), size=K)]
    if sum(counts.values()) != len(tr):
        raise AssertionError('six mutually exclusive relaxation classes must reconcile')
    tr['match_stratum'] = labels
    tr['candidate_count'] = np.asarray(sizes, dtype=np.int64)
    indices = tr.trigger_idx.to_numpy(dtype=np.int64)
    tr['trigger_month'] = ctx.index[indices].strftime('%Y-%m')
    tr['trigger_utc_hour'] = hour[indices]
    tr['trigger_atr_band'] = band[indices] + 1
    tr['trigger_atr_fraction'] = a[indices]
    tr['stop_distance_atr'] = (tr.e_ref - tr.sl).abs() / tr.atr15
    if len(tr) and not (np.isfinite(tr.stop_distance_atr).all() and (tr.stop_distance_atr > 0).all()):
        raise ValueError('actual trade stop distances must be finite and positive')
    return MatchDesign(tr, sample_indices, candidates, float(q1), float(q2), counts)


def _distribution(values):
    x = np.asarray(values, float)
    if not len(x):
        return {'n': 0, 'mean': None, 'population_sd': None, 'quantiles': {}}
    if not np.isfinite(x).all():
        raise ValueError('nonfinite balance diagnostic input')
    levels = [0, .05, .25, .5, .75, .95, 1]
    return {'n': len(x), 'mean': float(np.mean(x)), 'population_sd': float(np.std(x)),
            'quantiles': {str(p): float(q) for p, q in zip(levels, np.quantile(x, levels))}}


def _balance(ctx, design):
    keep = design.sample_indices[:, 0] >= 0
    actual = design.records.loc[keep]
    drawn = design.sample_indices[keep].ravel()
    d = np.repeat(actual.stop_distance_atr.to_numpy(), K)
    actual_i = actual.trigger_idx.to_numpy(dtype=np.int64)
    values = {
        'atr_fraction': (ctx.atr['atr_at'][actual_i] / ctx.C[actual_i],
                         ctx.atr['atr_at'][drawn] / ctx.C[drawn]),
        'stop_distance_bps': (np.abs(actual.e_ref.to_numpy() - actual.sl.to_numpy()) / actual.e_ref.to_numpy() * 1e4,
                              d * ctx.atr['atr_at'][drawn] / ctx.C[drawn] * 1e4),
    }
    result = {name: {'actual': _distribution(x), 'random': _distribution(y)}
              for name, (x, y) in values.items()}
    result['utc_hour'] = {
        'actual': np.bincount(np.asarray(ctx.index[actual_i].hour), minlength=24).tolist(),
        'random': np.bincount(np.asarray(ctx.index[drawn].hour), minlength=24).tolist(),
    }
    result['matched_actual_count'] = int(keep.sum())
    result['unmatched_actual_count'] = int((~keep).sum())
    return result


def _month_mean(values, entry_times):
    x = np.asarray(values, float)
    if not len(x):
        return np.nan
    month = pd.DatetimeIndex(entry_times).strftime('%Y-%m').to_numpy()
    unique, inverse = np.unique(month, return_inverse=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, x)
    return float(sums.sum() / len(x))


def auxiliary_randomization(actual, random_net_bps, entry_times, version_index):
    """One frozen row-major (5000,N) draw from already simulated controls."""
    actual = np.asarray(actual, float)
    random_net_bps = np.asarray(random_net_bps, float)
    n = len(actual)
    if random_net_bps.shape != (n, K):
        raise ValueError('auxiliary null requires exactly K=100 controls per matched trade')
    if not n:
        return {'n': 0, 'draws': NULL_DRAWS, 'null_means': np.empty(0),
                'observed_mean_bps': None, 'p': 1.0, 'mcse': 0.0}
    if not (np.isfinite(actual).all() and np.isfinite(random_net_bps).all()):
        raise ValueError('auxiliary randomization accepts only complete matched trades')
    rng = np.random.default_rng(VS.SEED_BASE + 900 + version_index)
    j = rng.integers(0, K, size=(NULL_DRAWS, n))
    null = np.empty(NULL_DRAWS)
    # The entire frozen J matrix is generated before bounded arithmetic batches.
    for begin in range(0, NULL_DRAWS, 100):
        end = min(begin + 100, NULL_DRAWS)
        null[begin:end] = random_net_bps[np.arange(n)[None, :], j[begin:end]].mean(axis=1)
    observed = _month_mean(actual, entry_times)
    p = float(np.mean(null >= observed))
    return {'n': n, 'draws': NULL_DRAWS, 'null_means': null,
            'observed_mean_bps': observed, 'p': p,
            'mcse': float(np.sqrt(p * (1 - p) / NULL_DRAWS))}


def run_matched_controls(ctx: Context, spec: VS.VersionSpec,
                         trades_by_cost: Mapping[str, pd.DataFrame]) -> MatchedControlsResult:
    """Simulate the same sampled entries under all four frozen cost scenarios."""
    if set(trades_by_cost) != set(VS.COSTS):
        raise ValueError('matched controls require all four frozen cost ledgers')
    design = build_match_design(ctx, spec, trades_by_cost[VS.MAIN_COST])
    records = design.records.copy()
    keep = design.sample_indices[:, 0] >= 0
    draw_indices = design.sample_indices[keep].ravel()
    sides = np.repeat(records.loc[keep, 'side'].to_numpy(dtype=float), K)
    distances = np.repeat(records.loc[keep, 'stop_distance_atr'].to_numpy(), K)
    reference = ctx.C[draw_indices]
    stops = reference - sides * distances * ctx.atr['atr_at'][draw_indices]
    identity = ('trigger_idx', 'side', 'e_ref', 'sl', 'atr15')
    random_by_cost = {}
    for cost_name, cost in VS.COSTS.items():
        actual = _sort_trades(ctx, trades_by_cost[cost_name])
        if len(actual) != len(records) or any(
                not np.array_equal(actual[col].to_numpy(), records[col].to_numpy(), equal_nan=True)
                for col in identity):
            raise ValueError(f'cost ledger trade identity differs: {cost_name}')
        simulated = E.simulate_events(ctx, draw_indices, sides, stops, reference,
                                      spec.hold == 'fixed1', cost)
        values = np.full((len(records), K), np.nan)
        values[keep] = simulated.net_bps.to_numpy().reshape(int(keep.sum()), K)
        random_by_cost[cost_name] = values
        random_mean = np.full(len(records), np.nan)
        random_mean[keep] = values[keep].mean(axis=1)
        records[f'actual_{cost_name}'] = actual.net_bps.to_numpy()
        records[f'random_mean_{cost_name}'] = random_mean
        records[f'excess_{cost_name}'] = actual.net_bps.to_numpy() - random_mean
    records['excess_bps'] = records[f'excess_{VS.MAIN_COST}']
    null = auxiliary_randomization(records.loc[keep, f'actual_{VS.MAIN_COST}'].to_numpy(),
                                    random_by_cost[VS.MAIN_COST][keep], records.loc[keep, 'entry_time'],
                                    list(VS.VERSIONS).index(spec.name))
    return MatchedControlsResult(records, design.sample_indices, random_by_cost,
                                 design.relaxation_counts, _balance(ctx, design), null,
                                 {'q1': design.q1, 'q2': design.q2}, len(design.candidate_indices))

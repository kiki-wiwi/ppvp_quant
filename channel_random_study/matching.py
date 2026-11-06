"""Fixed offline monthly conditional schedules, constructed before performance.

This module consumes only causal opportunity snapshots and flags. Whole-month
candidate membership and parent counts make the schedules offline objects;
execution outcomes, returns and holding intervals are never matching inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from .config import LAST_ENTRY, MAIN_START, R_COUNT, R_VERSIONS, SCHEDULE_SEED

KEYS = ['month','utc_hour','vol_bin']
HOUR_NS = 3_600_000_000_000


@dataclass(frozen=True)
class ScheduleSet:
    candidates: pd.DataFrame
    parents: pd.DataFrame
    strata: pd.DataFrame
    schedules: tuple[pd.DataFrame, ...]
    metadata: dict
    counters: dict


def _time_hash(index: pd.DatetimeIndex) -> str:
    """Portable SHA256 of ascending UTC nanoseconds encoded as int64 little-endian."""
    values = np.asarray(index.as_unit('ns').asi8, dtype='<i8')
    return hashlib.sha256(values.tobytes(order='C')).hexdigest()


def _validated_grid(opportunities: pd.DataFrame) -> pd.DataFrame:
    required = {'entry_time','signal_time','eligible','is_parent',*KEYS}
    if not required.issubset(opportunities.columns):
        raise ValueError(f'missing opportunity fields: {sorted(required - set(opportunities.columns))}')
    idx = pd.DatetimeIndex(opportunities.index)
    if idx.tz is None or idx.hasnans or idx.has_duplicates:
        raise ValueError('opportunities require unique finite timezone-aware entry times')
    out = opportunities.copy()
    out.index = idx.tz_convert('UTC').as_unit('ns')
    out.index.name = 'entry_time'
    out = out.sort_index()
    for name in ('eligible','is_parent'):
        if out[name].isna().any() or not out[name].isin([True,False]).all():
            raise ValueError(f'{name} must contain boolean flags')
        out[name] = out[name].astype(bool)
    if (out.is_parent & ~out.eligible).any():
        raise ValueError('parents must be a subset of common eligible candidates')
    entries = pd.DatetimeIndex(out.entry_time)
    signals = pd.DatetimeIndex(out.signal_time)
    if entries.tz is None or signals.tz is None or entries.hasnans or signals.hasnans:
        raise ValueError('signal and entry columns require finite timezone-aware timestamps')
    if not entries.tz_convert('UTC').as_unit('ns').equals(out.index):
        raise ValueError('entry column and index must agree')
    if not np.array_equal(signals.as_unit('ns').asi8 + HOUR_NS, out.index.asi8):
        raise ValueError('entry_time must equal signal_time + 1 hour')
    if np.any(out.index.asi8 % HOUR_NS):
        raise ValueError('opportunities must lie on the UTC hourly grid')
    if not np.array_equal(out.month.to_numpy(), out.index.strftime('%Y-%m')):
        raise ValueError('month labels must use entry-time UTC month')
    if not np.array_equal(out.utc_hour.to_numpy(), out.index.hour):
        raise ValueError('hour labels must use entry-time UTC hour')
    if out.vol_bin.isna().any() or not out.vol_bin.isin(range(6)).all():
        raise ValueError('volatility bins must be integers 0 through 5')
    eligible_index = out.index[out.eligible]
    if len(eligible_index) and (eligible_index[0] < MAIN_START + pd.Timedelta(hours=1)
                                or eligible_index[-1] > LAST_ENTRY):
        raise ValueError('eligible entry outside registered signal/entry bounds')
    return out


def build_schedules(opportunities: pd.DataFrame) -> ScheduleSet:
    """Build all 200 schedules from U/P, using exactly one spawn(200) operation."""
    grid = _validated_grid(opportunities)
    candidates = grid.loc[grid.eligible].copy()
    parents = candidates.loc[candidates.is_parent].copy()
    groups = candidates.groupby(KEYS, sort=True, observed=True)
    strata = groups.agg(N=('is_parent','size'), n=('is_parent','sum')).reset_index()
    strata[['N','n']] = strata[['N','n']].astype(np.int64)
    if ((strata.n < 0) | (strata.n > strata.N)).any():
        raise ValueError('invalid stratum capacity: n must satisfy 0 <= n <= N')
    if int(strata.N.sum()) != len(candidates) or int(strata.n.sum()) != len(parents):
        raise ValueError('stratum candidate/parent totals do not conserve the grid')
    strata['saturated'] = strata.n.eq(strata.N) & strata.n.gt(0)
    ordered_groups = []
    for row in strata.itertuples(index=False):
        positions = np.sort(np.asarray(groups.indices[(row.month,row.utc_hour,row.vol_bin)], dtype=np.int64))
        if len(positions) != row.N:
            raise ValueError('stratum capacity differs from its candidate array')
        ordered_groups.append((positions,int(row.n)))
    children = np.random.SeedSequence(SCHEDULE_SEED).spawn(R_COUNT)
    schedules = []
    replicate_metadata = []
    for j,child in enumerate(children):
        rng = np.random.Generator(np.random.PCG64(child))
        selected = []
        for positions,n in ordered_groups:
            N = len(positions)
            if n == 0:
                continue
            if n == N:
                selected.append(positions)
            else:
                chosen = np.sort(rng.choice(N, size=n, replace=False))
                if len(chosen) != n or len(np.unique(chosen)) != n or np.any(chosen < 0) or np.any(chosen >= N):
                    raise ValueError('invalid without-replacement draw')
                selected.append(positions[chosen])
        positions = np.sort(np.concatenate(selected)) if selected else np.empty(0,dtype=np.int64)
        if len(positions) != len(parents) or len(np.unique(positions)) != len(positions):
            raise ValueError('schedule count mismatch or duplicate candidate')
        schedule = candidates.iloc[positions].copy()
        actual = schedule.groupby(KEYS, sort=True, observed=True).size()
        expected = strata.set_index(KEYS).n
        if not np.array_equal(actual.reindex(expected.index, fill_value=0).to_numpy(), expected.to_numpy()):
            raise ValueError('schedule does not preserve every stratum parent count')
        schedules.append(schedule)
        replicate_metadata.append(dict(version=R_VERSIONS[j], spawn_key=list(child.spawn_key),
                                       time_sha256=_time_hash(schedule.index), count=len(schedule)))
    nonsaturated = int(strata.loc[(strata.n > 0) & (strata.n < strata.N), 'n'].sum())
    saturated = int(strata.loc[strata.saturated, 'n'].sum())
    missing = int(parents.vol_bin.eq(5).sum())
    counters = dict(candidate_count=len(candidates), parent_count=len(parents), stratum_count=len(strata),
                    saturated_stratum_count=int(strata.saturated.sum()), saturated_parent_count=saturated,
                    nonsaturated_parent_count=nonsaturated, missing_vol_parent_count=missing,
                    nonsaturated_parent_fraction=nonsaturated/len(parents) if len(parents) else 0.,
                    finite_positive_vol_parent_fraction=(len(parents)-missing)/len(parents) if len(parents) else 0.)
    metadata = dict(seed=SCHEDULE_SEED, bit_generator='PCG64', count=R_COUNT,
                    time_encoding='ascending UTC ns int64 little-endian C-order',
                    candidate_time_sha256=_time_hash(candidates.index), parent_time_sha256=_time_hash(parents.index),
                    replicates=replicate_metadata)
    return ScheduleSet(candidates, parents, strata, tuple(schedules), metadata, counters)

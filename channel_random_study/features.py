"""Causal price-channel inputs for the independent unfiltered-channel study.

Only an already-authorized Context is accepted. There is no market I/O, volume
profile, compression history, volume-strength filter, or execution-state input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (ATR_LENGTH, INITIAL_ATR_MULTIPLIER, LAST_ENTRY, MAIN_START,
                     PROFILE_BARS, VOLUME_LOOKBACK)
from .scope import Context, require_synthetic

HOUR = pd.Timedelta(hours=1)
FOUR_HOURS = pd.Timedelta(hours=4)
MINUTE = pd.Timedelta(minutes=1)


@dataclass(frozen=True)
class Features:
    """Hours use open-time indices; references use available/close-time indices."""

    hours: pd.DataFrame
    references: pd.DataFrame


def _aggregate(minutes: pd.DataFrame, frequency: str, duration: pd.Timedelta) -> pd.DataFrame:
    """Retain empty calendar buckets and exclude the unfinished final bucket."""
    bars = minutes.resample(frequency, origin='epoch', closed='left', label='left').agg(
        open=('open','first'), high=('high','max'), low=('low','min'),
        close=('close','last'), volume=('volume','sum'), n_valid=('close','count'),
        archive_gap=('archive_gap','max'),
    )
    bars.index = bars.index.as_unit('ns')
    bars['available_time'] = bars.index + duration
    return bars.loc[bars.available_time <= minutes.index[-1] + MINUTE].copy()


def _atr(hours: pd.DataFrame) -> np.ndarray:
    """Baseline Wilder rule: only nonempty hours advance the recursive state."""
    values = np.full(len(hours), np.nan, dtype=np.float64)
    previous_close = current = np.nan
    initial_sum = 0.
    observations = 0
    for i, row in enumerate(hours.itertuples()):
        if row.n_valid:
            tr = row.high - row.low
            if np.isfinite(previous_close):
                tr = max(tr, abs(row.high - previous_close), abs(row.low - previous_close))
            previous_close = row.close
            observations += 1
            if observations <= ATR_LENGTH:
                initial_sum += tr
                if observations == ATR_LENGTH:
                    current = initial_sum / ATR_LENGTH
            else:
                current = ((ATR_LENGTH - 1) * current + tr) / ATR_LENGTH
        values[i] = current
    return values


def build_features(ctx: Context) -> Features:
    """Compute closed 1H/4H features, preserving the full UTC calendar grid."""
    require_synthetic(ctx)
    minutes = pd.DataFrame(dict(open=ctx.O, high=ctx.H, low=ctx.L, close=ctx.C,
                                volume=ctx.V, archive_gap=ctx.archive_gap), index=ctx.index)
    minutes.loc[ctx.bad, ['open','high','low','close','volume']] = np.nan
    hours = _aggregate(minutes, '1h', HOUR)
    hours['atr'] = _atr(hours)
    # Earlier gaps remain in the recursive ATR state even after local windows move on.
    hours['atr_has_archive_gap'] = hours.archive_gap.cummax().astype(bool)
    bars = _aggregate(minutes, '4h', FOUR_HOURS)
    refs = pd.DataFrame(index=pd.DatetimeIndex(bars.available_time).as_unit('ns'))
    refs.index.name = 'reference_time'
    rolling = bars.rolling(PROFILE_BARS, min_periods=PROFILE_BARS)
    refs['U'] = rolling.high.max().to_numpy()
    refs['L'] = rolling.low.min().to_numpy()
    refs['C'] = bars.close.to_numpy()
    refs['total_volume'] = rolling.volume.sum().to_numpy()
    refs['n_valid'] = rolling.n_valid.sum().to_numpy()
    starts = refs.index - PROFILE_BARS * FOUR_HOURS
    refs['window_start'] = starts
    refs['window_complete'] = ((rolling.n_valid.min().to_numpy() > 0) & (starts >= ctx.index[0]))
    refs['window_valid'] = (refs.window_complete & np.isfinite(refs.U) & np.isfinite(refs.L)
                            & (refs.L > 0) & (refs.U > refs.L)
                            & np.isfinite(refs.C) & (refs.C > 0)
                            & np.isfinite(refs.total_volume) & (refs.total_volume > 0))
    refs['window_has_archive_gap'] = (bars.archive_gap.astype(int)
                                      .rolling(PROFILE_BARS, min_periods=1).max().to_numpy() > 0)
    return Features(hours, refs)


def build_opportunities(ctx: Context, features: Features | None = None) -> pd.DataFrame:
    """Return the hourly U/P audit grid indexed by planned entry time e=h+1H.

    `eligible` is common candidate membership. `is_parent` adds the crossing
    condition. Neither membership consults entry-minute status or future prices.
    The matching volatility uses the prior hour; the stop uses the signal hour.
    """
    require_synthetic(ctx)
    built = build_features(ctx) if features is None else features
    hours = built.hours.loc[built.hours.available_time <= ctx.index[-1] + MINUTE]
    out = hours.copy()
    out['signal_time'] = hours.index
    out['entry_time'] = hours.index + HOUR
    # This is the last 4H close available at h, not at e.
    out['reference_time'] = hours.index.floor('4h')
    chosen = built.references.reindex(pd.DatetimeIndex(out.reference_time))
    for column in chosen.columns:
        name = 'reference_n_valid' if column == 'n_valid' else column
        out[name] = chosen[column].to_numpy()
    for column in ('window_complete','window_valid','window_has_archive_gap'):
        out[column] = out[column].eq(True)
    out['boundary'] = out.U
    out['previous_close'] = hours.close.shift(1)
    out['previous_atr'] = hours.atr.shift(1)
    out['eref'] = hours.close
    out['sl'] = out.eref - INITIAL_ATR_MULTIPLIER * out.atr
    out['signal_hour_full'] = hours.n_valid.eq(60)
    out['hour_history_full'] = (hours.n_valid.eq(60).astype(int).shift(1)
                                .rolling(VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK)
                                .sum().eq(VOLUME_LOOKBACK))
    out['hour_lookback_has_archive_gap'] = (hours.archive_gap.astype(int)
                                           .rolling(VOLUME_LOOKBACK + 1, min_periods=1).max().gt(0))
    out['signal_lookback_has_archive_gap'] = (out.window_has_archive_gap
                                             | out.hour_lookback_has_archive_gap | out.atr_has_archive_gap)
    out['crossover'] = (out.previous_close <= out.boundary) & (out.eref > out.boundary)
    out['prevol'] = out.previous_atr / out.previous_close
    v = out.prevol.to_numpy(dtype=np.float64)
    bins = np.searchsorted(np.array([.0025,.005,.01,.02], dtype=np.float64), v, side='right')
    bins[~np.isfinite(v) | (v <= 0)] = 5
    out['vol_bin'] = bins.astype(np.int8)
    out['month'] = out.entry_time.dt.strftime('%Y-%m')
    out['utc_hour'] = out.entry_time.dt.hour
    reasons = [
        (out.signal_time < MAIN_START, 'before_main_start'),
        (out.entry_time > LAST_ENTRY, 'after_last_entry'),
        (~out.signal_hour_full, 'incomplete_signal_hour'),
        (~out.hour_history_full, 'incomplete_hour_history'),
        (~out.window_valid | ~np.isfinite(out.boundary), 'invalid_reference'),
        (~np.isfinite(out.eref) | ~np.isfinite(out.previous_close), 'invalid_close'),
        (~np.isfinite(out.atr), 'atr_uninitialized'),
        (out.atr <= 0, 'nonpositive_atr'),
        (~np.isfinite(out.sl) | (out.sl <= 0), 'nonpositive_initial_stop'),
    ]
    out['skip_reason'] = np.select([condition.to_numpy(dtype=bool) for condition,_ in reasons],
                                    [reason for _,reason in reasons], default='')
    out['eligible'] = out.skip_reason.eq('')
    out['is_parent'] = out.eligible & out.crossover
    out.index = pd.DatetimeIndex(out.entry_time).as_unit('ns')
    out.index.name = 'entry_time'
    return out

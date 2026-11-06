"""Causal trend/dip features; all inputs pass this study's independent gate.

No input files, execution prices, account state, or future minute are inspected.
The borrowed aggregation and Wilder routines are pure, read-only utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, localcontext
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

from .config import (DIP_CHANGE_ATR, DIP_MEAN_ATR, FAST_SMA, HOLD_MINUTES,
                     LAST_DECISION, LIMIT_ATR, MAIN_START, MEAN_LOOKBACK,
                     PRICE_TICK, SLOW_SMA, STOP_ATR)
from .scope import require_synthetic

if TYPE_CHECKING:
    from .scope import Context

HOUR = pd.Timedelta(hours=1)
MINUTE = pd.Timedelta(minutes=1)


@dataclass(frozen=True)
class Features:
    """Hours use opening time; references use (timeframe, available time)."""

    hours: pd.DataFrame
    references: pd.DataFrame


def build_features(ctx: Context) -> Features:
    """Keep UTC calendar buckets and advance Wilder state only on nonempty hours."""
    require_synthetic(ctx)
    from breakout_study.features import _aggregate, _atr

    minutes = pd.DataFrame({'open': ctx.O, 'high': ctx.H, 'low': ctx.L,
                            'close': ctx.C, 'volume': ctx.V,
                            'archive_gap': ctx.archive_gap}, index=ctx.index)
    minutes.loc[ctx.bad, ['open', 'high', 'low', 'close', 'volume']] = np.nan
    hours = _aggregate(minutes, '1h', HOUR)
    hours['atr_after_hour'] = _atr(hours)
    # Wilder retains all prior observations; its input-gap flag remains cumulative.
    hours['atr_has_archive_gap'] = hours.archive_gap.cummax().astype(bool)
    references = []
    for frequency, duration in [('1D', pd.Timedelta(days=1)), ('4h', 4 * HOUR)]:
        # UTC 24-hour ticks are the natural daily grid, without pandas'
        # non-Tick '1D' origin warning. The public timeframe stays '1D'.
        bars = _aggregate(minutes, '24h' if frequency == '1D' else frequency, duration)
        refs = bars.copy()
        refs['open_time'] = bars.index
        refs.index = pd.DatetimeIndex(bars.available_time).as_unit('ns')
        refs.index.name = 'reference_time'
        refs['sma20'] = bars.close.rolling(FAST_SMA, min_periods=FAST_SMA).mean().to_numpy()
        refs['sma50'] = bars.close.rolling(SLOW_SMA, min_periods=SLOW_SMA).mean().to_numpy()
        refs['trend'] = ((refs.close > refs.sma20) & (refs.sma20 > refs.sma50)
                         & np.isfinite(refs[['close', 'sma20', 'sma50']]).all(axis=1))
        refs['window_has_archive_gap'] = (bars.archive_gap.astype(int)
            .rolling(SLOW_SMA, min_periods=1).max().gt(0).to_numpy())
        references.append(refs)
    combined = pd.concat(references, keys=['1D', '4h'], names=['timeframe', 'reference_time'])
    return Features(hours, combined)



def floor_price_offset(eref: float, atr: float, multiplier: float) -> float:
    """Apply the registered floor using the inputs' canonical decimal values.

    Eref=103.3 and ATR=2 therefore gives S=99.3, without binary division
    accidentally returning 99.2. Values strictly below a tick still round down;
    there is no tolerance or promotion of near-boundary values. Decimal
    precision is local so another module cannot alter this calculation.
    """
    if not (np.isfinite(eref) and np.isfinite(atr)):
        return np.nan
    with localcontext() as arithmetic:
        arithmetic.prec = 50
        tick = Decimal(str(PRICE_TICK))
        offset = Decimal(str(eref)) - Decimal(str(multiplier)) * Decimal(str(atr))
        return float((offset / tick).to_integral_value(rounding=ROUND_FLOOR) * tick)


def build_signals(ctx: Context, features: Features | None = None) -> pd.DataFrame:
    """Return every completed hour, including rejected parent-signal decisions.

    Higher-timeframe references are fixed at hour start h. ATR and the 20-hour
    mean also stop before h; the signal's own close is observed at d = h + 1h.
    Entry-minute post-only and maintenance checks belong to the execution engine.
    """
    require_synthetic(ctx)
    built = build_features(ctx) if features is None else features
    hours = built.hours.loc[built.hours.available_time <= ctx.index[-1] + MINUTE]
    out = hours.copy()
    out['signal_time'] = hours.index
    out['decision_time'] = hours.index + HOUR
    out['deadline'] = out.decision_time + HOLD_MINUTES * MINUTE
    out['parent_id'] = out.decision_time.dt.strftime('%Y%m%dT%H%M%SZ')
    out['eref'] = hours.close
    out['atr'] = hours.atr_after_hour.shift(1)
    out['previous_close'] = hours.close.shift(1)
    out['target'] = hours.close.shift(1).rolling(MEAN_LOOKBACK, min_periods=MEAN_LOOKBACK).mean()
    out['sl'] = [floor_price_offset(e, a, STOP_ATR) for e, a in zip(out.eref, out.atr)]
    out['limit'] = [floor_price_offset(e, a, LIMIT_ATR) for e, a in zip(out.eref, out.atr)]
    out['signal_hour_full'] = hours.n_valid.eq(60)
    out['mean_history_full'] = (hours.n_valid.eq(60).astype(int).shift(1)
        .rolling(MEAN_LOOKBACK, min_periods=MEAN_LOOKBACK).sum().eq(MEAN_LOOKBACK))
    for frequency, prefix in [('1D', 'daily'), ('4h', 'four_hour')]:
        reference_times = hours.index.floor(frequency)
        out[f'{prefix}_reference_time'] = reference_times
        refs = built.references.loc[frequency] if frequency in built.references.index.get_level_values(0) else pd.DataFrame()
        chosen = refs.reindex(reference_times)
        for name in ['close', 'sma20', 'sma50', 'trend', 'window_has_archive_gap']:
            values = chosen[name] if name in chosen else pd.Series(np.nan, index=reference_times)
            if name in {'trend', 'window_has_archive_gap'}:
                values = values.eq(True)
            out[f'{prefix}_{name}'] = values.to_numpy()
    out['dip_mean'] = out.eref <= out.target - DIP_MEAN_ATR * out.atr
    out['dip_change'] = out.eref <= out.previous_close - DIP_CHANGE_ATR * out.atr
    out['signal_lookback_has_archive_gap'] = (
        hours.archive_gap.astype(int).rolling(MEAN_LOOKBACK + 1, min_periods=1).max().gt(0)
        | hours.atr_has_archive_gap.shift(1).eq(True)
        | out.daily_window_has_archive_gap | out.four_hour_window_has_archive_gap)
    valid_prices = (np.isfinite(out[['eref', 'atr', 'target', 'sl', 'limit', 'previous_close']]).all(axis=1)
                    & (out.sl > 0) & (out.sl < out.limit)
                    & (out.limit < out.eref) & (out.eref < out.target))
    reasons = [
        (out.signal_time < MAIN_START, 'before_main_start'),
        (out.decision_time > LAST_DECISION, 'after_last_decision'),
        (~out.signal_hour_full, 'incomplete_signal_hour'),
        (~out.mean_history_full, 'incomplete_mean_history'),
        (~np.isfinite(out.atr), 'atr_uninitialized'),
        (out.atr <= 0, 'nonpositive_atr'),
        (~out.daily_trend | ~out.four_hour_trend, 'not_bullish_trend'),
        (~out.dip_mean | ~out.dip_change, 'no_dip'),
        (~valid_prices, 'invalid_order_prices'),
    ]
    out['skip_reason'] = np.select([mask.to_numpy(dtype=bool) for mask, _ in reasons],
                                    [reason for _, reason in reasons], default='')
    out['eligible'] = out.skip_reason.eq('')
    out.index = pd.DatetimeIndex(out.decision_time).as_unit('ns')
    out.index.name = 'decision_time'
    return out

"""Closed-bar features for the fixed channel-versus-VAH preregistration.

This module accepts an already authorized Context and performs no market I/O.
All windows retain their UTC calendar grid, including wholly bad buckets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from impulse_study.daily import profile_levels

from .scope import require_synthetic
from .config import (
    ATR_LENGTH, COMPRESSION_HISTORY, COMPRESSION_QUANTILE,
    INITIAL_ATR_MULTIPLIER, LAST_ENTRY, MAIN_START, PROFILE_BARS,
    PROFILE_BINS, VALUE_AREA, VERSIONS, VOLUME_LOOKBACK, VOLUME_MULTIPLIER,
)

if TYPE_CHECKING:
    from .scope import Context

HOUR = pd.Timedelta(hours=1)
FOUR_HOURS = pd.Timedelta(hours=4)
MINUTE = pd.Timedelta(minutes=1)


@dataclass(frozen=True)
class Features:
    """Hours indexed by open time; references indexed by 4H close time."""

    hours: pd.DataFrame
    references: pd.DataFrame


def _utc_index(index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(index, utc=True)).as_unit("ns")


def _aggregate(minutes: pd.DataFrame, frequency: str, duration: pd.Timedelta) -> pd.DataFrame:
    columns = ["open", "high", "low", "close", "volume", "n_valid", "archive_gap", "available_time"]
    if minutes.empty:
        out = pd.DataFrame({c: pd.Series(dtype=float) for c in columns},
                           index=pd.DatetimeIndex([], tz="UTC").as_unit("ns"))
        out["archive_gap"] = out.archive_gap.astype(bool)
        out["available_time"] = pd.Series([], index=out.index, dtype="datetime64[ns, UTC]")
        return out
    out = minutes.resample(frequency, origin="epoch", closed="left", label="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        n_valid=("close", "count"), archive_gap=("archive_gap", "max"),
    )
    out.index = _utc_index(out.index)
    out["available_time"] = out.index + duration
    return out.loc[out.available_time <= minutes.index[-1] + MINUTE].copy()


def _atr(hours: pd.DataFrame) -> np.ndarray:
    """Wilder state advances only on nonempty hours, including partial hours."""
    out = np.full(len(hours), np.nan)
    previous_close = np.nan
    current_atr = np.nan
    initial_sum = 0.0
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
                    current_atr = initial_sum / ATR_LENGTH
            else:
                current_atr = ((ATR_LENGTH - 1) * current_atr + tr) / ATR_LENGTH
        out[i] = current_atr
    return out


def build_features(ctx: Context) -> Features:
    """Compute the approved features without observing unfinished candles."""
    require_synthetic(ctx)
    idx = _utc_index(ctx.index)
    minutes = pd.DataFrame({"open": ctx.O, "high": ctx.H, "low": ctx.L,
                            "close": ctx.C, "volume": ctx.V,
                            "archive_gap": ctx.archive_gap}, index=idx)
    price_columns = ["open", "high", "low", "close", "volume"]
    minutes.loc[np.asarray(ctx.bad, dtype=bool), price_columns] = np.nan
    hours = _aggregate(minutes, "1h", HOUR)
    hours["atr"] = _atr(hours)
    # Wilder ATR is recursive: even a remote flagged input remains part of its state.
    hours["atr_has_archive_gap"] = hours.archive_gap.cummax().astype(bool)
    bars4 = _aggregate(minutes, "4h", FOUR_HOURS)
    refs = pd.DataFrame(index=_utc_index(bars4.available_time))
    refs.index.name = "reference_time"
    rolling = bars4.rolling(PROFILE_BARS, min_periods=PROFILE_BARS)
    refs["U"] = rolling.high.max().to_numpy()
    refs["L"] = rolling.low.min().to_numpy()
    refs["C"] = bars4.close.to_numpy()
    refs["total_volume"] = rolling.volume.sum().to_numpy()
    refs["n_valid"] = rolling.n_valid.sum().to_numpy()
    has_all_buckets = rolling.n_valid.min().to_numpy() > 0
    starts = refs.index - PROFILE_BARS * FOUR_HOURS
    coverage = starts >= idx[0] if len(idx) else np.zeros(len(refs), dtype=bool)
    refs["window_valid"] = (has_all_buckets & coverage &
                            (refs.total_volume > 0) & (refs.U > refs.L) &
                            np.isfinite(refs.C) & (refs.C > 0))
    refs["width"] = ((refs.U - refs.L) / refs.C).where(refs.window_valid)
    refs["compression_q"] = (refs.width.shift(1)
                              .rolling(COMPRESSION_HISTORY, min_periods=COMPRESSION_HISTORY)
                              .quantile(COMPRESSION_QUANTILE, interpolation="linear"))
    refs["compressed"] = (refs.width <= refs.compression_q) & np.isfinite(refs.compression_q)
    # A flag records observed source gaps; it never discards an otherwise valid bar.
    refs["window_has_archive_gap"] = (bars4.archive_gap.astype(int)
                                      .rolling(PROFILE_BARS, min_periods=1).max().to_numpy() > 0)
    refs["width_history_has_archive_gap"] = (refs.window_has_archive_gap.astype(int).shift(1)
                                             .rolling(COMPRESSION_HISTORY, min_periods=1).max()
                                             .fillna(0).astype(bool))
    for column in ("POC", "VAL", "VAH"):
        refs[column] = np.nan
    timestamps = idx.asi8
    good = ~np.asarray(ctx.bad, dtype=bool)
    for position in np.flatnonzero(refs.window_valid.to_numpy()):
        end = refs.index[position]
        start = end - PROFILE_BARS * FOUR_HOURS
        left, right = np.searchsorted(timestamps, [start.value, end.value])
        valid = good[left:right]
        poc, val, vah, _, _ = profile_levels(
            np.asarray(ctx.L)[left:right][valid], np.asarray(ctx.H)[left:right][valid],
            np.asarray(ctx.V)[left:right][valid], n_bins=PROFILE_BINS, va_pct=VALUE_AREA,
        )
        refs.iloc[position, refs.columns.get_indexer(["POC", "VAL", "VAH"])] = [poc, val, vah]
    return Features(hours=hours, references=refs)


def build_signals(ctx: Context, version: str, features: Features | None = None) -> pd.DataFrame:
    """Return every closed-hour decision, including explicit non-entry reasons.

    Eligibility is known at the hour close. Availability, bad status, and price
    of the entry minute are intentionally left to the execution engine.
    """
    require_synthetic(ctx)
    if version not in VERSIONS:
        raise ValueError(f"unregistered version: {version!r}")
    built = build_features(ctx) if features is None else features
    idx = _utc_index(ctx.index)
    hours = built.hours
    if len(idx):
        hours = hours.loc[hours.available_time <= idx[-1] + MINUTE]
    else:
        hours = hours.iloc[:0]
    out = hours.copy()
    out["signal_time"] = hours.index
    out["entry_time"] = hours.index + HOUR
    out["reference_time"] = hours.index.floor("4h")
    chosen = built.references.reindex(pd.DatetimeIndex(out.reference_time))
    for column in chosen.columns:
        # Hour valid count remains n_valid; reference count is exposed separately.
        name = "reference_n_valid" if column == "n_valid" else column
        out[name] = chosen[column].to_numpy()
    out["boundary"] = out["U" if version == "P_channel" else "VAH"]
    out["previous_close"] = hours.close.shift(1)
    out["eref"] = hours.close
    out["sl"] = out.eref - INITIAL_ATR_MULTIPLIER * out.atr
    out["baseline_volume"] = (hours.volume.shift(1)
                               .rolling(VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).median())
    out["signal_hour_full"] = hours.n_valid.eq(60)
    out["volume_history_full"] = (hours.n_valid.eq(60).astype(int).shift(1)
                                   .rolling(VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).sum()
                                   .eq(VOLUME_LOOKBACK))
    out["volume_ok"] = (out.signal_hour_full & out.volume_history_full &
                         (out.baseline_volume > 0) &
                         (out.volume >= VOLUME_MULTIPLIER * out.baseline_volume))
    out["crossover"] = (out.previous_close <= out.boundary) & (out.eref > out.boundary)
    for column in ("window_valid", "compressed", "window_has_archive_gap", "width_history_has_archive_gap"):
        out[column] = out[column].eq(True)
    out["volume_lookback_has_archive_gap"] = (hours.archive_gap.astype(int)
                                              .rolling(VOLUME_LOOKBACK + 1, min_periods=1).max().gt(0))
    out["signal_lookback_has_archive_gap"] = (out.window_has_archive_gap |
                                               out.width_history_has_archive_gap |
                                               out.volume_lookback_has_archive_gap |
                                               out.atr_has_archive_gap)
    reasons = [
        (out.signal_time < MAIN_START, "before_main_start"),
        (out.entry_time > LAST_ENTRY, "after_last_entry"),
        (~out.signal_hour_full, "incomplete_signal_hour"),
        (~out.volume_history_full, "incomplete_volume_history"),
        (~out.window_valid | ~np.isfinite(out.boundary), "invalid_reference"),
        (~np.isfinite(out.compression_q), "insufficient_width_history"),
        (~out.compressed, "not_compressed"),
        (out.baseline_volume <= 0, "nonpositive_volume_baseline"),
        (~out.volume_ok, "insufficient_volume"),
        (~out.crossover, "no_crossover"),
        (~np.isfinite(out.atr), "atr_uninitialized"),
        (out.atr <= 0, "nonpositive_atr"),
        (~np.isfinite(out.sl) | (out.sl <= 0), "nonpositive_initial_stop"),
    ]
    out["skip_reason"] = np.select([condition.to_numpy(dtype=bool) for condition, _ in reasons],
                                    [reason for _, reason in reasons], default="")
    out["eligible"] = out.skip_reason.eq("")
    out.index = pd.DatetimeIndex(out.entry_time).as_unit("ns")
    out.index.name = "entry_time"
    return out

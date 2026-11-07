"""Causal daily trend and shared ex ante risk budget."""
import numpy as np
import pandas as pd

from . import config, scope

DAY = pd.Timedelta(days=1)
MINUTE = pd.Timedelta(minutes=1)
MINUTES_PER_DAY = 1440


def build_features(ctx: scope.Context) -> pd.DataFrame:
    """Index features by executable midnight; audit fields refer to the prior UTC day."""
    scope.require_synthetic(ctx)
    first_day = ctx.index[0].ceil('D')
    last_boundary = (ctx.index[-1] + MINUTE).floor('D')
    start = ctx.index.searchsorted(first_day)
    end = ctx.index.searchsorted(last_boundary)
    if last_boundary < first_day:
        end = start

    # A UTC day counts only when its entire minute grid is present. A day
    # with some bad minutes still contributes its last valid close.
    valid = ~ctx.bad[start:end].reshape(-1, MINUTES_PER_DAY)
    valid_counts = valid.sum(axis=1)
    if (valid_counts == 0).any():
        missing = first_day + int(np.flatnonzero(valid_counts == 0)[0]) * DAY
        raise ValueError(f'UTC day has no valid minutes: {missing}')
    closes = ctx.C[start:end].reshape(-1, MINUTES_PER_DAY)
    last_valid = MINUTES_PER_DAY - 1 - valid[:, ::-1].argmax(axis=1)
    daily_close = closes[np.arange(len(closes)), last_valid]
    if not np.isfinite(daily_close).all() or (daily_close <= 0).any():
        raise ValueError('completed daily closes must be finite and positive')
    daily_gap = ctx.archive_gap[start:end].reshape(-1, MINUTES_PER_DAY).any(axis=1)

    days = pd.date_range(max(config.MAIN_START, first_day), ctx.index[-1].floor('D'),
                         freq='D', name='decision_time').as_unit('ns')
    prior = np.asarray((days - first_day).days, dtype=np.intp) - 1
    if (prior < config.TREND_DAYS - 1).any():
        raise ValueError('each decision requires 60 completed UTC days')

    eref = daily_close[prior]
    trend_offsets = np.arange(1 - config.TREND_DAYS, 1)
    sma = daily_close[prior[:, None] + trend_offsets].mean(axis=1)
    vol_offsets = np.arange(1 - config.VOL_DAYS, 1)
    return_days = prior[:, None] + vol_offsets
    with np.errstate(over='ignore', invalid='ignore'):
        returns = daily_close[return_days] / daily_close[return_days - 1] - 1.
        sigma = np.sqrt(365.) * returns.std(axis=1, ddof=1)
    if not np.isfinite(sma).all() or not np.isfinite(returns).all() or not np.isfinite(sigma).all():
        raise ValueError('daily trend and volatility inputs must be finite')

    updates = np.asarray((days == config.MAIN_START) | (days.weekday == 0))
    held_budget = np.full(len(days), np.nan)
    held_budget[updates] = config.LEVERAGE_CAP
    limited = updates & (sigma > config.TARGET_VOL / config.LEVERAGE_CAP)
    held_budget[limited] = config.TARGET_VOL / sigma[limited]
    budget = pd.Series(held_budget, index=days).ffill()
    if not np.isfinite(budget).all() or (budget <= 0).any():
        raise ValueError('weekly budget must be finite and positive')

    return pd.DataFrame({
        'eref': eref, 'sma60': sma, 'sigma': sigma, 'budget': budget,
        'g': eref > sma, 'budget_update': updates,
        'n_valid_prior': valid_counts[prior], 'has_archive_gap': daily_gap[prior],
    }, index=days)

"""Invented full-calendar price and funding fixture."""
import numpy as np
import pandas as pd

from . import config, scope


def study_context():
    days = pd.date_range(config.IS_START, config.IS_END, freq='D', inclusive='left')
    day = np.arange(len(days))
    # Alternating drift and volatility produce on/off regimes and changing budgets.
    drift = np.where(day // 69 % 2 == 0, .003, -.003)
    amplitude = np.where(day // 91 % 2 == 0, .017, .004)
    daily = 10_000. * np.exp(np.cumsum(drift + amplitude * np.sin(day * .91)))
    preceding = np.r_[10_000., daily[:-1]]
    minutes_per_day = 1440
    fractions = np.arange(1, minutes_per_day + 1) / minutes_per_day
    close = (preceding[:, None] + (daily - preceding)[:, None] * fractions).ravel()
    opened = np.r_[10_000., close[:-1]]
    index = pd.date_range(config.IS_START, config.IS_END, freq='min', inclusive='left').as_unit('ns')
    bad = np.zeros(len(index), dtype=bool)
    gaps = np.zeros(len(index), dtype=bool)
    for when in ('2020-03-02 00:00', '2021-05-07 13:30'):
        location = index.get_loc(pd.Timestamp(when, tz='UTC'))
        bad[location] = True
        gaps[location] = True

    frame = pd.DataFrame({
        'open': opened, 'high': np.maximum(opened, close) + 1.,
        'low': np.minimum(opened, close) - 1., 'close': close,
        'volume': np.where(bad, 0., 100.), 'bad': bad, 'archive_gap': gaps,
    }, index=index, copy=False)
    frame.attrs['provenance'] = 'synthetic'
    settlements = pd.date_range(config.IS_START, config.IS_END, freq='8h', inclusive='left')
    rates = 0.00003 + 0.00005 * np.sin(np.arange(len(settlements)) * .4)
    funding = pd.Series(rates, index=settlements)
    return scope.make_context(frame, funding)

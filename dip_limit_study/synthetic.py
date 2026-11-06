"""Deterministic invented trend/dips for integration; never reads input files."""
import numpy as np
import pandas as pd
from .config import IS_START, IS_END
from .scope import make_context


def study_context():
    """Full IS calendar with four invented intraday dips per calendar month.

    All numbers below describe a test path, not observations or fitted strategy
    parameters. Smooth daily/4H closing trends coexist with dips at noon and a
    second-minute penetration before recovery. No random or real input is used.
    """
    index = pd.date_range(IS_START, IS_END, freq='min', inclusive='left').as_unit('ns')
    baseline = 10_000. + np.arange(len(index)) * .001
    close = baseline.copy()
    event_slices = []
    count = 0
    for month in pd.date_range(IS_START, IS_END, freq='MS', inclusive='left'):
        for day in (3, 10, 17, 24):
            h = index.get_loc(month + pd.Timedelta(days=day - 1, hours=12))
            d = h + 60
            depth = 30. + 3. * (count % 7)
            close[h:d] = np.linspace(baseline[h], baseline[d - 1] - depth, 60)
            eref = close[d - 1]
            close[d] = eref - .25
            close[d + 1] = eref - 3.85
            close[d + 2:d + 60] = np.linspace(eref - 3., baseline[d + 59], 58)
            event_slices.append((h, d + 60))
            count += 1
    opened = np.r_[close[0], close[:-1]]
    high = np.maximum(opened, close) + 1.
    low = np.minimum(opened, close) - 1.
    # Narrow event-minute wicks preserve a hand-checkable penetration/stop gap.
    for start, end in event_slices:
        high[start:end] = np.maximum(opened[start:end], close[start:end]) + .05
        low[start:end] = np.minimum(opened[start:end], close[start:end]) - .05
    frame = pd.DataFrame({'open': opened, 'high': high, 'low': low, 'close': close,
                          'volume': 10_000., 'bad': False, 'archive_gap': False}, index=index)
    frame.attrs['provenance'] = 'synthetic'
    funding = pd.Series(.00001, index=pd.date_range(IS_START, IS_END, freq='8h', inclusive='left'))
    return make_context(frame, funding)

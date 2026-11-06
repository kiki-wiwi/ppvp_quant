"""Approved constants; no parameter-search or command-line overrides."""
from dataclasses import dataclass
from types import MappingProxyType
import pandas as pd

IS_START = pd.Timestamp('2020-01-01', tz='UTC')
MAIN_START = pd.Timestamp('2020-03-01', tz='UTC')
IS_END = pd.Timestamp('2024-01-01', tz='UTC')
LAST_ENTRY = pd.Timestamp('2023-12-28 23:00', tz='UTC')
HALF = pd.Timestamp('2022-01-01', tz='UTC')
INITIAL_CAPITAL = 100_000.
RISK_FRACTION = .005
MIN_DIST_FRAC = .0015
LEVERAGE_CAP = 2.
HOLD_MINUTES = 4320
COOLDOWN_MINUTES = 240
VERSIONS = ('P_channel', 'V_vah')
PROFILE_BARS = 30
COMPRESSION_HISTORY = 180
COMPRESSION_QUANTILE = .25
PROFILE_BINS = 50
VALUE_AREA = .70
VOLUME_LOOKBACK = 20
VOLUME_MULTIPLIER = 1.5
ATR_LENGTH = 14
INITIAL_ATR_MULTIPLIER = 2.
TRAIL_ATR_MULTIPLIER = 3.


@dataclass(frozen=True)
class Cost:
    name: str
    fee: float
    entry_slip: float
    exit_slip: float
    include_funding: bool

    @property
    def c_rt(self):
        return 2*self.fee + self.entry_slip + self.exit_slip


COSTS = MappingProxyType({
    'gross_0': Cost('gross_0', 0., 0., 0., False),
    'main_17': Cost('main_17', .0005, .0005, .0002, True),
    'harsh_34': Cost('harsh_34', .001, .001, .0004, True),
})

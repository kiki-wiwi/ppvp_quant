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
S_VERSION = 'S_channel_unfiltered'
R_COUNT = 200
R_VERSIONS = tuple(f'R_{j:03d}' for j in range(R_COUNT))
VERSIONS = (S_VERSION,) + R_VERSIONS
PROFILE_BARS = 30
VOLUME_LOOKBACK = 20
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

SCHEDULE_SEED = 2026110901
WEBB_SEED = 2026110902
MONTH_SEED = 2026110903
WEBB_DRAWS = 99_999
MONTH_DRAWS = 10_000
N_MONTHS = 46
N_DAYS = 1401
BASELINE = "366c592c40a9f33ace51f600e7f5bc04ff2bd6f3"
APPROVED_PLAN_SHA256 = "ad8a451a6afb059691b5f3b0ed96f98d2f28390086492bffb3b87419f8a942a0"

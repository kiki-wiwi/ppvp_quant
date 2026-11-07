"""Approved study constants; no runtime parameter search or overrides."""
from dataclasses import dataclass
from types import MappingProxyType
import pandas as pd

BASELINE = '35490be840ca49ec0d7172e4356ae556df94f640'
APPROVED_PLAN_SHA256 = '1d7a2684a53bb64d8cc323f9eb2a1ffb652d1ff1ecb5f54ef38d51e460f8546f'
IS_START = pd.Timestamp('2020-01-01', tz='UTC')
MAIN_START = pd.Timestamp('2020-03-01', tz='UTC')
IS_END = pd.Timestamp('2024-01-01', tz='UTC')
HALF = pd.Timestamp('2022-01-01', tz='UTC')
INITIAL_CAPITAL = 100_000.
TREND_DAYS = 60
VOL_DAYS = 20
TARGET_VOL = .20
LEVERAGE_CAP = 1.
T_VERSION = 'T_daily_trend'
P_VERSION = 'P_continuous_long'
VERSIONS = (T_VERSION, P_VERSION)
N_DAYS = 1401
N_MONTHS = 46
WEBB_SEED = 2026111001
MONTH_SEED = 2026111002
WEBB_DRAWS = 99_999
MONTH_DRAWS = 10_000

@dataclass(frozen=True)
class Cost:
    name: str
    fee: float
    buy_slip: float
    sell_slip: float
    include_funding: bool

    @property
    def c_rt(self):
        return 2*self.fee + self.buy_slip + self.sell_slip

COSTS = MappingProxyType({
    'gross_0': Cost('gross_0', 0., 0., 0., False),
    'main_17': Cost('main_17', .0005, .0005, .0002, True),
    'harsh_34': Cost('harsh_34', .001, .001, .0004, True),
})

"""Approved constants, with no runtime parameter-search overrides."""
from dataclasses import dataclass
from types import MappingProxyType
import pandas as pd

IS_START=pd.Timestamp('2020-01-01',tz='UTC')
MAIN_START=pd.Timestamp('2020-03-01',tz='UTC')
IS_END=pd.Timestamp('2024-01-01',tz='UTC')
LAST_DECISION=pd.Timestamp('2023-12-30 23:00',tz='UTC')
HALF=pd.Timestamp('2022-01-01',tz='UTC')
INITIAL_CAPITAL=100_000.
RISK_FRACTION=.005
MIN_DIST_FRAC=.0015
LEVERAGE_CAP=2.
PRICE_TICK=.10
QTY_STEP=.001
HOLD_MINUTES=1440
TTL_MINUTES=60
COOLDOWN_MINUTES=240
ATR_LENGTH=14
FAST_SMA=20
SLOW_SMA=50
MEAN_LOOKBACK=20
DIP_MEAN_ATR=2.
DIP_CHANGE_ATR=1.
STOP_ATR=2.
LIMIT_ATR=.5
STRESS_DELAY_MINUTES=1
STRESS_PENETRATION=.0002
VERSIONS=('M_market','L_limit')

@dataclass(frozen=True)
class Cost:
    taker: float
    maker: float
    entry_slip: float
    exit_slip: float
    budget: float
    include_funding: bool

COSTS=MappingProxyType({
    'gross':Cost(0.,0.,0.,0.,0.,False),
    'main':Cost(.0005,.0002,.0005,.0002,.0017,True),
    'harsh':Cost(.001,.0004,.001,.0004,.0034,True),
})

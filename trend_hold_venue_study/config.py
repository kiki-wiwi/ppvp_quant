"""Approved fixed constants and evaluation windows."""
from dataclasses import dataclass
from types import MappingProxyType
import pandas as pd
from trend_hold_study.config import Cost, COSTS

BASELINE = '3846971ad9014043cc0371a96c693924b7ca1e61'
INITIAL_CAPITAL = 100_000.
TREND_DAYS = 60
VOL_DAYS = 20
TARGET_VOL = .20
LEVERAGE_CAP = 1.
CASH_EPS = 1e-9
MMR = .005
WEBB_SEED = 2026111001
MONTH_SEED = 2026111002
WEBB_DRAWS = 99_999
MONTH_DRAWS = 10_000
DAY_MINUTES = 1440
VERSIONS = ('V1_USDM','P_USDM','V2_SPOT','P_SPOT',
            'V3_CM_DeltaMatched','P_CM_DeltaMatched','V4_CM_CoinAccum','HODL_1x_Coin')
CM_VERSIONS = VERSIONS[4:]
VENUES = ('UM','SPOT','CM')
ASSETS = ('BTC','ETH')
CONTRACT_SIZE = MappingProxyType({'BTC':100.,'ETH':10.})
DECLARATION = 'User-authorized one-shot descriptive OOS and cross-venue evaluation; R5 failed V3/V5; no automatic promotion.'

def utc(value):
    value = pd.Timestamp(value)
    if value.tz is None:
        raise ValueError('timezone-aware timestamp required')
    return value.tz_convert('UTC').as_unit('ns')

@dataclass(frozen=True)
class Window:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    signal_origin: pd.Timestamp
    def __post_init__(self):
        for k in ('start','end','signal_origin'):
            object.__setattr__(self,k,utc(getattr(self,k)))
        if self.start >= self.end or self.signal_origin > self.start:
            raise ValueError('invalid window')
        if any(getattr(self,k) != getattr(self,k).normalize() for k in ('start','end','signal_origin')):
            raise ValueError('window boundaries must be UTC midnight')

def window(name, asset='BTC'):
    if asset not in ASSETS: raise ValueError('unregistered asset')
    definitions = {
        'IS-Full': ('2020-03-01','2024-01-01','2020-03-01'),
        'IS-Common': ('2020-09-01','2024-01-01','2020-03-01' if asset=='BTC' else '2020-08-31'),
        'IS-CM-SelfSignal': ('2020-11-01','2024-01-01','2020-11-01'),
        'Post-IS': ('2024-01-01','2026-11-01','2024-01-01'),
    }
    if name not in definitions or name=='IS-Full' and asset!='BTC':
        raise ValueError('unregistered asset/window')
    a,b,c=definitions[name]
    return Window(name,*(pd.Timestamp(x,tz='UTC') for x in (a,b,c)))

"""
impulse_study/scope.py — 本研究轮次的数据范围常量与越界断言。

本轮只允许使用 BTCUSDT 永续 IS [2020-01-01, 2024-01-01) 的数据，以及 2020-01-01 之前的现货预热数据。
任何读取入口在返回数据前调用这里的断言；越界即抛出 ScopeError，而不是静默截断。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import IS_END, IS_START

SYMBOL = "BTCUSDT"
IS_MONTHS = ("2020-01", "2023-12")              # 永续 1m / 15m / 1d 官方月度归档：只下载这 48 个月
SPOT_1D_WARMUP_MONTHS = ("2019-09", "2019-12")  # ATR_1D 价格预热（只用价格，不用成交量）
SPOT_1M_WARMUP_MONTHS = ("2019-11", "2019-12")  # 仅供“现货成交量拼接预热”敏感性计数（S4），不进入主口径
MINUTE = pd.Timedelta(minutes=1)
DAY = pd.Timedelta(days=1)
MINUTE_NS = 60 * 1_000_000_000
DAY_NS = 86_400 * 1_000_000_000
IS_START_NS = int(IS_START.value)
IS_END_NS = int(IS_END.value)


class ScopeError(RuntimeError):
    """读取或产出了本轮范围之外的数据（VAL / OOS / ETH，或预热数据越过 IS 起点）。"""


def _ns(idx) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(idx).as_unit("ns").asi8, dtype=np.int64)


def assert_within_is(idx, what: str) -> None:
    """全部时间戳落在 [IS_START, IS_END)。"""
    v = _ns(idx)
    if len(v) and (v.min() < IS_START_NS or v.max() >= IS_END_NS):
        raise ScopeError(f"{what}: timestamps [{pd.Timestamp(v.min(), tz='UTC')}, {pd.Timestamp(v.max(), tz='UTC')}] "
                         f"leave the IS window [{IS_START}, {IS_END})")


def assert_before_is(idx, what: str) -> None:
    """全部时间戳早于 IS_START（预热数据）。"""
    v = _ns(idx)
    if len(v) and v.max() >= IS_START_NS:
        raise ScopeError(f"{what}: warm-up data reaches {pd.Timestamp(v.max(), tz='UTC')} >= {IS_START}")


def assert_symbol(symbol: str) -> None:
    if symbol != SYMBOL:
        raise ScopeError(f"symbol {symbol} is outside this round's scope (BTCUSDT only)")

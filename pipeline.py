"""
pipeline.py — 数据 → 4H PPVP 箱体 → 1H 状态机信号 → 回测 → 分段指标 的统一入口（带箱体缓存与 OOS 锁）。

``pipeline.run`` 是 Phase 2/3 唯一受支持的研究入口。研究阶段默认 ``end = RESEARCH_DATA_END (= 2025-07-01)``：
OOS 区间的 K 线根本不会被送入引擎。只有在 Phase 4 锁定最终参数后，才允许以 ``allow_oos=True`` 运行到 2026-11-01。
OOS 锁在每个入口（load_market / get_boxes / run）缓存查找之前检查；缓存对象以浅拷贝返回（pandas 3 写时复制），
调用方就地修改返回的 DataFrame 不会污染缓存。
"""
from __future__ import annotations

import pandas as pd

from backtester import run_backtest
from config import (BASELINE_BACKTEST, BASELINE_STRATEGY, OOS_END, RESEARCH_DATA_END, RESEARCH_START,
                    BacktestConfig, StrategyConfig)
from data_loader import load_funding, load_klines
from metrics import OOSLockedError, segment_report
from ppvp_engine import build_boxes
from strategy_state_machine import generate_signals

_MARKET: dict = {}
_BOXES: dict = {}


def clear_cache() -> None:
    _MARKET.clear()
    _BOXES.clear()


def _check_lock(end: pd.Timestamp, allow_oos: bool) -> None:
    if end > RESEARCH_DATA_END and not allow_oos:
        raise OOSLockedError(f"data beyond {RESEARCH_DATA_END} is locked until final parameters are frozen")


def spot_daily_warmup(symbol: str) -> pd.Series:
    """2020 年之前的现货日线收盘（仅供慢速价格类过滤器预热）。"""
    s = load_klines(symbol, "1h", "spot")
    d = s["close"].resample("1D").last().dropna()
    return d[d.index < RESEARCH_START]


def load_market(symbol: str, end: pd.Timestamp = RESEARCH_DATA_END, allow_oos: bool = False):
    """返回 (df1h, funding, spot_daily)。1H 取 open_time < end；资金费取结算时刻 ≤ end
    （end 恰为最后一根 1H 的收盘，该时刻的结算由 end 之前的溢价决定，收盘市价离场需要承担它）。"""
    _check_lock(end, allow_oos)
    key = (symbol, end)
    if key not in _MARKET:
        df = load_klines(symbol, "1h", "perp")
        df = df[(df.index >= RESEARCH_START) & (df.index < end)]
        fund = load_funding(symbol)
        fund = fund[(fund.index >= RESEARCH_START) & (fund.index <= end)]
        _MARKET[key] = (df, fund, spot_daily_warmup(symbol))
    df, fund, spot = _MARKET[key]
    return df.copy(deep=False), fund.copy(deep=False), spot.copy(deep=False)


def get_boxes(symbol: str, scfg: StrategyConfig, end: pd.Timestamp = RESEARCH_DATA_END, allow_oos: bool = False):
    _check_lock(end, allow_oos)
    key = (symbol, end, scfg.structural_key())
    if key not in _BOXES:
        df1h, _, _ = load_market(symbol, end, allow_oos)
        _BOXES[key] = build_boxes(df1h, scfg)
    boxes, df4h = _BOXES[key]
    return boxes.copy(deep=False), df4h.copy(deep=False)


def run(symbol: str = "BTCUSDT", scfg: StrategyConfig = BASELINE_STRATEGY, bcfg: BacktestConfig = BASELINE_BACKTEST,
        end: pd.Timestamp = RESEARCH_DATA_END, segments=("IS", "VAL"), allow_oos: bool = False) -> dict:
    df1h, fund, spot_d = load_market(symbol, end, allow_oos)
    boxes, df4h = get_boxes(symbol, scfg, end, allow_oos)
    sig = generate_signals(df1h, scfg, boxes, df4h, spot_d)
    res = run_backtest(df1h, sig, bcfg, fund)
    rep = segment_report(res, segments, allow_oos) if segments else {}
    return {"df1h": df1h, "df4h": df4h, "boxes": boxes, "signals": sig, "result": res, "report": rep}


def run_full_period(symbol: str, scfg: StrategyConfig, bcfg: BacktestConfig) -> dict:
    """Phase 4 专用：在参数锁定后运行完整 2020-01-01 ~ 2026-11-01。"""
    return run(symbol, scfg, bcfg, end=OOS_END, segments=("IS", "VAL", "OOS"), allow_oos=True)

"""
pipeline.py — 数据 → 4H PPVP 箱体 → 1H 状态机信号 → 回测 → 分段指标 的统一入口（带箱体缓存与 OOS 锁）。

研究阶段默认 ``end = RESEARCH_DATA_END (= 2025-07-01)``：OOS 区间的 K 线根本不会被送入引擎。
只有在 Phase 4 锁定最终参数后，才允许以 ``allow_oos=True`` 运行到 2026-11-01。
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


def spot_daily_warmup(symbol: str) -> pd.Series:
    """2020 年之前的现货日线收盘（仅供慢速价格类过滤器预热）。"""
    s = load_klines(symbol, "1h", "spot")
    d = s["close"].resample("1D").last().dropna()
    return d[d.index < RESEARCH_START]


def load_market(symbol: str, end: pd.Timestamp = RESEARCH_DATA_END, allow_oos: bool = False):
    if end > RESEARCH_DATA_END and not allow_oos:
        raise OOSLockedError(f"data beyond {RESEARCH_DATA_END} is locked until final parameters are frozen")
    key = (symbol, end)
    if key not in _MARKET:
        df = load_klines(symbol, "1h", "perp")
        df = df[(df.index >= RESEARCH_START) & (df.index < end)]
        fund = load_funding(symbol)
        fund = fund[(fund.index >= RESEARCH_START) & (fund.index < end)]
        _MARKET[key] = (df, fund, spot_daily_warmup(symbol))
    return _MARKET[key]


def get_boxes(symbol: str, scfg: StrategyConfig, end: pd.Timestamp = RESEARCH_DATA_END, allow_oos: bool = False):
    key = (symbol, end, scfg.structural_key())
    if key not in _BOXES:
        df1h, _, _ = load_market(symbol, end, allow_oos)
        _BOXES[key] = build_boxes(df1h, scfg)
    return _BOXES[key]


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

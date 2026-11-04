"""
metrics.py — 绩效指标（日度净值口径）与样本分段统计。

* 净值先重采样为 UTC 每日收盘净值（日 D 的收盘 = D+1 00:00 时刻的盯市净值；空仓日自然为 0 收益），
  ``Sharpe = mean(daily_ret) / std(daily_ret, ddof=1) * sqrt(365)``，无风险利率 0。严禁按每笔交易收益算夏普。
* Sortino 使用下行偏差 sqrt(mean(min(r, 0)^2))；Calmar = CAGR / |MaxDD|（日度净值口径）。
* OOS 隔离：请求 'OOS' 分段必须显式传入 ``allow_oos=True``（仅 Phase 4 锁定参数后使用）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester import BacktestResult
from config import SEGMENTS

ANN = 365.0


class OOSLockedError(RuntimeError):
    pass


def daily_equity(res: BacktestResult) -> pd.Series:
    """UTC 日度收盘净值，索引为日期 D（值 = D+1 00:00 的净值）；首行为起始日前一天的初始资金。

    显式按 ``(close_time - 1ns).floor('D')`` 归属日期后取当日最后一个净值，
    不使用 ``resample(closed='right')``（pandas 3 会在末尾多生成一个空桶，ffill 后形成虚假的 0 收益日）。
    """
    s = pd.concat([pd.Series([res.initial_capital], index=[res.start_time]), res.equity])
    lab = (s.index - pd.Timedelta(1, "ns")).floor("D")
    d = s.groupby(lab).last()
    full = pd.date_range(d.index[0], d.index[-1], freq="D")
    return d.reindex(full).ffill()


def daily_returns(d_eq: pd.Series) -> pd.Series:
    return d_eq.pct_change().iloc[1:]


def sharpe(r: pd.Series) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd and sd > 0 else float("nan")


def sortino(r: pd.Series) -> float:
    dd = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
    return float(r.mean() / dd * np.sqrt(ANN)) if dd > 0 else float("nan")


def max_drawdown(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1.0).min())


def drawdown_series(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1.0


def trade_stats(tr: pd.DataFrame) -> dict:
    closed = tr[tr["exit_reason"] != "OPEN"] if len(tr) else tr
    n = int(len(closed))
    if n == 0:
        return {"trades": 0, "win_rate": np.nan, "profit_factor": np.nan, "avg_r": np.nan, "avg_hold_h": np.nan,
                "avg_win_pct": np.nan, "avg_loss_pct": np.nan, "fees": 0.0, "funding": 0.0}
    pnl = closed["pnl"]
    gw, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "trades": n,
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(gw / gl) if gl > 0 else float("inf"),
        "avg_r": float(closed["r_multiple"].mean()),
        "avg_hold_h": float(closed["hold_bars"].mean()),
        "avg_win_pct": float(closed.loc[pnl > 0, "ret_on_equity"].mean()) if (pnl > 0).any() else np.nan,
        "avg_loss_pct": float(closed.loc[pnl < 0, "ret_on_equity"].mean()) if (pnl < 0).any() else np.nan,
        "fees": float(closed["fees"].sum()),
        "funding": float(closed["funding"].sum()),
        "exit_reasons": closed["exit_reason"].value_counts().to_dict(),
    }


def segment_metrics(res: BacktestResult, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """[start, end) 分段：日度收益、夏普等风险指标 + 该段内入场的交易统计。"""
    d_eq = daily_equity(res)
    base_day = start - pd.Timedelta(days=1)
    seg = d_eq[(d_eq.index >= base_day) & (d_eq.index < end)]
    if len(seg) < 2 or seg.index[0] != base_day:
        raise ValueError(f"segment [{start}, {end}) not covered by backtest equity")
    r = daily_returns(seg)
    days = len(r)
    total = seg.iloc[-1] / seg.iloc[0] - 1.0
    cagr = (seg.iloc[-1] / seg.iloc[0]) ** (ANN / days) - 1.0
    mdd = max_drawdown(seg)
    hourly = res.equity[(res.equity.index > start) & (res.equity.index <= end)]
    pos = res.position[(res.position.index > start) & (res.position.index <= end)]
    tr = res.trades
    if len(tr):
        tr = tr[(tr["entry_time"] >= start) & (tr["entry_time"] < end)]
    out = {
        "start": str(start.date()), "end": str((end - pd.Timedelta(days=1)).date()), "days": days,
        "sharpe": sharpe(r), "sortino": sortino(r), "cagr": float(cagr), "total_return": float(total),
        "max_dd": mdd, "max_dd_hourly": max_drawdown(pd.concat([seg.iloc[:1], hourly])) if len(hourly) else mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else float("nan"),
        "exposure": float((pos > 0).mean()) if len(pos) else float("nan"),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(ANN)),
    }
    out.update(trade_stats(tr))
    return out


def segment_report(res: BacktestResult, segments=("IS", "VAL"), allow_oos: bool = False) -> dict:
    if "OOS" in segments and not allow_oos:
        raise OOSLockedError("OOS metrics are locked until final parameters are frozen (Phase 4).")
    return {name: segment_metrics(res, *SEGMENTS[name]) for name in segments}


def monthly_returns(d_eq: pd.Series) -> pd.DataFrame:
    m = d_eq.resample("ME").last()
    first = d_eq.iloc[:1]
    m = pd.concat([first, m])
    r = m.pct_change().iloc[1:]
    r = r[~r.index.duplicated(keep="last")]
    tbl = pd.DataFrame({"year": r.index.year, "month": r.index.month, "ret": r.to_numpy()})
    return tbl.pivot(index="year", columns="month", values="ret")

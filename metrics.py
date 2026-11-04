"""
metrics.py — 绩效指标（日度净值口径）与样本分段统计。

口径（均为显式约定，由 test_audit_regressions.py 钉住）
------------------------------------------------------
* 日度净值：日 D 的收盘净值 = D+1 00:00 时刻的盯市净值（按 ``(close_time - 1ns).floor('D')`` 归属）；
  起点锚 = 初始资金，标注在 ``start_time.floor('D') - 1 天``。空仓日自然为 0 收益。
* ``Sharpe = mean(daily_ret) / std(daily_ret, ddof=1) * sqrt(365)``，无风险利率 0。严禁按每笔交易收益算夏普。
* Sortino = mean / sqrt(mean(min(r, 0)^2)) * sqrt(365)（目标 0 的总体下行偏差，分母为全部 N 天）。
* CAGR = (E_end / E_start)^(365 / 自然日数) - 1；Calmar = 分段 CAGR / |分段日度 MaxDD|（非滚动 36 个月）。
* max_dd 为日收盘净值回撤；max_dd_hourly 为逐小时收盘盯市回撤（不含 K 线内的不利波动，后者见交易表 mae_pct）。
* compounding=False 的配置以初始资金为固定基数：分段净值按 初始资金 + 段内累计盈亏 重新定基。
  只有当分段内没有入场被净值上限截断（capped_entries == 0）时，VAL 的收益率/回撤才与 IS 期末净值无关；
  上限截断时（永续 leverage_cap*净值；现货 净值/(1+taker)）仓位由当期净值决定，capped_entries 报告其笔数。
* 分段边界必须是 UTC 零点；分段净值必须完整覆盖 [start, end)，否则报错（不会把截断数据当作完整分段）。
* 交易统计按 entry_time ∈ [start, end) 归属分段（跨段交易的全部盈亏计入入场分段，trades_open_at_end 报告其数量）；
  exit_reason='OPEN' 的期末未平仓交易不计入胜率等统计。pnl == 0 计为非盈利；无亏损交易时盈亏比为 inf（全为 0 时为 NaN）。
* OOS 隔离：任何触及 [OOS_START, ...) 的分段指标或月度收益、以及对净值延伸到 OOS_START 之后的回测结果
  计算任何分段指标，都必须显式传入 ``allow_oos=True``（仅 Phase 4 锁定参数后使用）。后者防止在全量结果上
  统计 VAL 时，把 VAL 入场、OOS 期间离场的交易盈亏计入 VAL。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester import BacktestResult
from config import OOS_START, SEGMENTS

ANN = 365.0
DAY = pd.Timedelta(days=1)


class OOSLockedError(RuntimeError):
    pass


def daily_equity(res: BacktestResult) -> pd.Series:
    """UTC 日度收盘净值，索引为日期 D（值 = D+1 00:00 的净值）；首行为起始日前一天的初始资金锚。

    显式按 ``(close_time - 1ns).floor('D')`` 归属日期后取当日最后一个净值，
    不使用 ``resample(closed='right')``（pandas 3 会在末尾多生成一个空桶，ffill 后形成虚假的 0 收益日）。
    锚点标注在 ``start_time.floor('D') - 1 天``：午夜起点与此前一致；非午夜起点不会被当日收盘覆盖。
    """
    eq = res.equity
    lab = (eq.index - pd.Timedelta(1, "ns")).floor("D")
    d = eq.groupby(lab).last()
    anchor = pd.Series([res.initial_capital], index=[res.start_time.floor("D") - DAY])
    d = pd.concat([anchor, d])
    d = d[~d.index.duplicated(keep="last")]
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


_EMPTY_STATS = {"trades": 0, "win_rate": np.nan, "profit_factor": np.nan, "avg_r": np.nan, "avg_hold_bars": np.nan,
                "avg_win_pct": np.nan, "avg_loss_pct": np.nan, "fees": 0.0, "funding": 0.0, "exit_reasons": {},
                "capped_entries": 0, "funding_incomplete": 0}


def trade_stats(tr: pd.DataFrame) -> dict:
    """已平仓交易的统计；capped_entries / funding_incomplete 按全部入场（含期末未平仓）计数。"""
    entries = {"capped_entries": int(tr["capped"].sum()) if len(tr) and "capped" in tr.columns else 0,
               "funding_incomplete": int((~tr["funding_complete"]).sum()) if len(tr) and "funding_complete" in tr.columns else 0}
    closed = tr[tr["exit_reason"] != "OPEN"] if len(tr) else tr
    n = int(len(closed))
    if n == 0:
        return dict(_EMPTY_STATS, exit_reasons={}, **entries)
    pnl = closed["pnl"]
    ret = closed["ret_on_base"] if "ret_on_base" in closed.columns else closed["ret_on_equity"]
    gw, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    pf = float(gw / gl) if gl > 0 else (float("inf") if gw > 0 else float("nan"))
    out = {
        "trades": n,
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": pf,
        "avg_r": float(closed["r_multiple"].mean()),
        "avg_hold_bars": float(closed["hold_bars"].mean()),
        "avg_win_pct": float(ret[pnl > 0].mean()) if (pnl > 0).any() else np.nan,
        "avg_loss_pct": float(ret[pnl < 0].mean()) if (pnl < 0).any() else np.nan,
        "fees": float(closed["fees"].sum()),
        "funding": float(closed["funding"].sum()),
        "exit_reasons": {str(k): int(v) for k, v in closed["exit_reason"].value_counts().items()},
    }
    out.update(entries)
    return out


def _check_oos(end: pd.Timestamp, allow_oos: bool) -> None:
    if end > OOS_START and not allow_oos:
        raise OOSLockedError("metrics on or after OOS_START are locked until final parameters are frozen (Phase 4).")


def segment_metrics(res: BacktestResult, start: pd.Timestamp, end: pd.Timestamp, allow_oos: bool = False) -> dict:
    """[start, end) 分段（UTC 零点边界）：日度收益、夏普等风险指标 + 该段内入场的交易统计。"""
    _check_oos(end, allow_oos)
    if not allow_oos and len(res.equity) and res.equity.index[-1] > OOS_START:
        raise OOSLockedError("this backtest result extends beyond OOS_START; segment metrics on it require allow_oos=True")
    if start != start.floor("D") or end != end.floor("D") or not end > start:
        raise ValueError(f"segment bounds must be UTC midnights with end > start: [{start}, {end})")
    if len(res.equity) == 0 or res.equity.index[-1] < end:
        raise ValueError(f"segment [{start}, {end}) not covered: equity ends before {end}")
    d_eq = daily_equity(res)
    base_day, last_day = start - DAY, end - DAY
    seg = d_eq[(d_eq.index >= base_day) & (d_eq.index <= last_day)]
    if len(seg) < 2 or seg.index[0] != base_day or seg.index[-1] != last_day:
        raise ValueError(f"segment [{start}, {end}) not covered by backtest equity")
    hourly = res.equity[(res.equity.index > start) & (res.equity.index <= end)]
    if not res.config.compounding:                     # 固定资金基数：按段内累计盈亏重新定基
        shift = res.initial_capital - seg.iloc[0]
        seg = seg + shift
        hourly = hourly + shift
    r = daily_returns(seg)
    days = len(r)
    total = seg.iloc[-1] / seg.iloc[0] - 1.0
    cagr = (seg.iloc[-1] / seg.iloc[0]) ** (ANN / days) - 1.0
    mdd = max_drawdown(seg)
    pos = res.position[(res.position.index > start) & (res.position.index <= end)]
    tr = res.trades
    if len(tr):
        tr = tr[(tr["entry_time"] >= start) & (tr["entry_time"] < end)]
    open_at_end = int(((tr["exit_reason"] == "OPEN") | (tr["exit_time"] > end)).sum()) if len(tr) else 0
    out = {
        "start": str(seg.index[1].date()), "end": str(seg.index[-1].date()), "days": days,
        "sharpe": sharpe(r), "sortino": sortino(r), "cagr": float(cagr), "total_return": float(total),
        "max_dd": mdd, "max_dd_hourly": max_drawdown(pd.concat([seg.iloc[:1], hourly])) if len(hourly) else mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else float("nan"),
        "exposure": float((pos > 0).mean()) if len(pos) else float("nan"),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(ANN)),
        "trades_open_at_end": open_at_end,
    }
    out.update(trade_stats(tr))
    return out


def segment_report(res: BacktestResult, segments=("IS", "VAL"), allow_oos: bool = False) -> dict:
    if "OOS" in segments and not allow_oos:
        raise OOSLockedError("OOS metrics are locked until final parameters are frozen (Phase 4).")
    return {name: segment_metrics(res, *SEGMENTS[name], allow_oos=allow_oos) for name in segments}


def monthly_returns(d_eq: pd.Series, allow_oos: bool = False) -> pd.DataFrame:
    """月度收益表（行 = 年，列 = 月）。首个月份为起点锚之后的第一个真实月份。"""
    if len(d_eq) and d_eq.index[-1] >= OOS_START and not allow_oos:
        raise OOSLockedError("monthly returns reaching OOS are locked until final parameters are frozen (Phase 4).")
    m = pd.concat([d_eq.iloc[:1], d_eq.resample("ME").last()])
    m = m[~m.index.duplicated(keep="first")]
    r = m.pct_change().iloc[1:]
    tbl = pd.DataFrame({"year": r.index.year, "month": r.index.month, "ret": r.to_numpy()})
    return tbl.pivot(index="year", columns="month", values="ret")

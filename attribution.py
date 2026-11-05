"""
attribution.py — 基线与实验的研究诊断（Phase 2 起使用）。

内容：交易级派生特征（几何、成本、严格 MFE/MAE、结构属性）、入场时刻的因果市场状态特征、
同几何括号单的随机入场零假设（全局 / 局部时间窗）、日度回撤区间、分桶统计与夏普的平稳自助法置信区间。

口径
----
* R 倍数统一以 风险 = qty * (entry_px - sl) 为分母（entry_px 含滑点），与 backtester 的 r_multiple 一致。
* 严格 MFE/MAE 只取入场 K 线之后、离场 K 线之前的完整 K 线（entry_bar+1 .. exit_bar-1）：离场 K 线内
  高低点的先后顺序未知，计入它会把“止损前曾到达 POC”之类的判断建立在不可知的 K 线内路径上。
* 随机入场零假设：保持每笔真实交易的止损距离 d_sl 与止盈距离 d_tp（相对入场成交价）不变，只把入场时刻换成
  随机 1H 收盘，按与回测器相同的规则撮合（止损优先、止损成交 min(open, SL)、止盈按 TP 成交、单边滑点、
  入场与止损 taker、止盈 maker、不计资金费）。比较量是不含资金费的 R（真实交易用 (gross_pnl - fees) / 风险）。
  bracket_outcomes 在真实入场 K 线上必须逐笔复现回测器的离场 K 线、原因与不含资金费的 R（test_attribution.py）。
* 市场状态特征只使用决策时刻（信号 K 线收盘）之前已收盘的数据：日线 D 于 D+1 00:00 可用，4H K 线于
  available_time 可用，1H 收益取到信号 K 线收盘为止（截断测试见 test_attribution.py）。

OOS 隔离：trade_frame / segment_trades / bucket_stats / random_entry_null 检查交易表的入场与离场时刻，
regime_features / bracket_outcomes / random_entry_null 检查实际参与计算的 K 线范围，drawdown_episodes /
stationary_bootstrap_sharpe / paired_sharpe_diff / segment_daily_returns / buy_and_hold 检查序列的时间范围；
任何一项延伸到 OOS_START 之后时须显式传入 allow_oos=True（仅 Phase 4 参数锁定后使用）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester import BacktestResult
from config import OOS_START, BacktestConfig, StrategyConfig
from metrics import (ANN, DAY, OOSLockedError, _check_result_extent, _check_series_extent, _check_trades_extent,
                     daily_equity, daily_returns, max_drawdown, sharpe, sortino)
from numba_compat import njit
from ppvp_engine import shift_known_at_open, to_ns, wilder_atr
from strategy_state_machine import daily_closes_with_warmup

NS_HOUR = np.int64(3600 * 10**9)
NS_DAY = np.int64(86400 * 10**9)


def _check_bars_extent(df1h: pd.DataFrame, n_used: int, allow_oos: bool) -> None:
    """参与计算的前 n_used 根 1H 的最后收盘晚于 OOS_START 时，必须显式 allow_oos=True。"""
    if allow_oos or n_used <= 0:
        return
    if df1h.index[n_used - 1] + pd.Timedelta(hours=1) > OOS_START:
        raise OOSLockedError("1H bars beyond OOS_START enter this computation; it requires allow_oos=True")


# ---------------------------------------------------------------------------
# 交易级特征
# ---------------------------------------------------------------------------
def trade_frame(res: BacktestResult, df1h: pd.DataFrame, boxes: pd.DataFrame, scfg: StrategyConfig,
                allow_oos: bool = False) -> pd.DataFrame:
    """已平仓交易 + 派生特征（只支持 tp_mode='vah' 的全平离场；分批离场的严格 MFE 定义不同）。"""
    _check_result_extent(res, allow_oos)
    bcfg: BacktestConfig = res.config
    tr = res.trades
    tf = tr[tr["exit_reason"] != "OPEN"].copy().reset_index(drop=True)
    if len(tf) == 0:
        return tf
    if (tf["partial_bar"] >= 0).any():
        raise ValueError("trade_frame supports full exits only (no POC partials)")
    h = df1h["high"].to_numpy(dtype=np.float64)
    l = df1h["low"].to_numpy(dtype=np.float64)
    c = df1h["close"].to_numpy(dtype=np.float64)
    slip = float(bcfg.slippage)
    eb = tf["entry_bar"].to_numpy(dtype=np.int64)
    xb = tf["exit_bar"].to_numpy(dtype=np.int64)
    ep, xp, q = (tf[k].to_numpy(dtype=np.float64) for k in ("entry_px", "exit_px", "qty"))
    sl, tp, poc = (tf[k].to_numpy(dtype=np.float64) for k in ("sl", "tp", "poc"))

    risk_unit = ep - sl
    tf["risk_usd"] = q * risk_unit
    tf["d_sl"] = 1.0 - sl / ep
    tf["d_tp"] = tp / ep - 1.0
    tf["rr_fill"] = (tp - ep) / risk_unit
    tf["p_rw"] = tf["d_sl"] / (tf["d_sl"] + tf["d_tp"])          # 无漂移连续随机游走下先触及止盈的概率
    # 成交价 = 参考价 * (1 ± 滑点)，因此滑点成本可由成交价精确反推
    tf["slip_usd"] = q * (ep - ep / (1.0 + slip)) + q * (xp / (1.0 - slip) - xp)
    tf["pre_cost_pnl"] = tf["gross_pnl"] + tf["slip_usd"]
    tf["cost_usd"] = tf["slip_usd"] + tf["fees"] + tf["funding"]
    tf["cost_r"] = tf["cost_usd"] / tf["risk_usd"]
    tf["r_ex_funding"] = (tf["gross_pnl"] - tf["fees"]) / tf["risk_usd"]
    tf["win"] = tf["pnl"] > 0

    # 严格 MFE / MAE（entry_bar+1 .. exit_bar-1；无完整 K 线时为 NaN）
    mfe_s = np.full(len(tf), np.nan)
    mae_s = np.full(len(tf), np.nan)
    for j in range(len(tf)):
        a, b = eb[j] + 1, xb[j]
        if b > a:
            mfe_s[j] = h[a:b].max()
            mae_s[j] = l[a:b].min()
    tf["mfe_r"] = (tf["max_high"] - ep) / risk_unit
    tf["mfe_r_strict"] = (mfe_s - ep) / risk_unit
    tf["mae_r_strict"] = (ep - mae_s) / risk_unit
    poc_above = poc > ep
    tf["poc_above_entry"] = poc_above
    tf["poc_r"] = (poc - ep) / risk_unit
    with np.errstate(invalid="ignore"):
        tf["reached_poc_strict"] = poc_above & (mfe_s >= poc)
        tf["reached_1r_strict"] = mfe_s >= ep + risk_unit
        tf["reached_half_r_strict"] = mfe_s >= ep + 0.5 * risk_unit
    # 离场 K 线同时触及 TP 与 SL：按规格悲观地判为止损
    tf["ambiguous_exit_bar"] = (tf["exit_reason"] == "SL").to_numpy() & (h[xb] >= tp)

    # 结构属性
    close_e = c[eb]
    atr = tf["atr"].to_numpy(dtype=np.float64)
    tf["width_atr"] = (tf["VAH"] - tf["VAL"]) / tf["atr"]
    tf["width_pct"] = (tf["VAH"] - tf["VAL"]) / tf["VAL"]
    tf["sl_atr"] = (close_e - sl) / atr
    tf["entry_vs_val_atr"] = (close_e - tf["VAL"].to_numpy()) / atr
    tf["entry_above_band"] = close_e > tf["zone_hi"].to_numpy()
    tf["touch_gap_bars"] = tf["t2_start"] - tf["t1_bar"]
    tf["pattern_lag_bars"] = tf["entry_bar"] - tf["t2_start"]
    tf["sfp"] = tf["t2_low"] < tf["t1_low"]
    tf["pattern_name"] = tf["pattern"].map({1: "pinbar", 2: "engulfing", 3: "both"})
    bx = boxes.loc[tf["box_id"].to_numpy()]
    age = pd.DatetimeIndex(tf["entry_time"]) - pd.DatetimeIndex(bx["available_time"])
    tf["box_age_h"] = np.asarray(age / pd.Timedelta(hours=1), dtype=np.float64)
    tf["box_hours"] = bx["n_1h"].to_numpy()
    tf["pivot_pair"] = (bx["pivot_a_type"].astype(str) + bx["pivot_b_type"].astype(str)).to_numpy()

    # 上一笔交易在同一支撑区且亏损（同区：|VAL 差| ≤ (k1+k2)*ATR，且间隔 < 7 天）
    ov = (scfg.zone_k1 + scfg.zone_k2) * atr
    prev_val = tf["VAL"].shift(1).to_numpy()
    prev_loss = (tf["pnl"].shift(1) < 0).to_numpy()
    gap_ok = (tf["entry_time"] - tf["exit_time"].shift(1)) < pd.Timedelta(days=7)
    tf["reentry_after_loss_same_zone"] = (np.abs(tf["VAL"].to_numpy() - prev_val) <= ov) & prev_loss & gap_ok.to_numpy()
    return tf


# ---------------------------------------------------------------------------
# 入场时刻的因果市场状态特征
# ---------------------------------------------------------------------------
def crash_event_times(df4h: pd.DataFrame, body_atr4: float = 2.0, vol_mult4: float = 2.0) -> np.ndarray:
    """4H 放量大阴线（实体 ≥ x*ATR_4H(14) 且 量 ≥ y*均量(20)，ATR/均量取前一根 4H 收盘值）的可用时刻（ns）。
    与 strategy_state_machine.regime_block_mask 的 crash_cooldown 定义相同。"""
    o4, h4, l4, c4, v4 = (df4h[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close", "volume"))
    atr4 = shift_known_at_open(wilder_atr(h4, l4, c4, 14))
    vs4 = shift_known_at_open(pd.Series(v4).rolling(20, min_periods=20).mean().to_numpy())
    with np.errstate(invalid="ignore"):
        crash = ((o4 - c4) >= body_atr4 * atr4) & (v4 >= vol_mult4 * vs4)
    return to_ns(df4h["available_time"])[crash]


def regime_features(df1h: pd.DataFrame, df4h: pd.DataFrame, spot_daily: pd.Series | None,
                    decision_bars: np.ndarray, allow_oos: bool = False) -> pd.DataFrame:
    """在第 i 根 1H 收盘（决策时刻）计算的市场状态特征；只使用该时刻之前已收盘的日线 / 4H / 1H 数据。

    above_sma200 / sma200_up（20 日斜率）/ mom20d / vol30d / vol30d_pct（相对过去 365 天）/ dd365（距 365 日高点）
    来自最近一根已收盘日线；hours_since_crash 来自 4H；ret24h / ret168h 来自 1H 收盘。
    df1h（全部用于日线与滚动计算）延伸到 OOS_START 之后时须 allow_oos=True。
    """
    _check_bars_extent(df1h, len(df1h), allow_oos)
    bars = np.asarray(decision_bars, dtype=np.int64)
    t_dec = to_ns(df1h.index)[bars] + NS_HOUR
    d = daily_closes_with_warmup(df1h, spot_daily)
    sma = d.rolling(200, min_periods=200).mean()
    lr = np.log(d).diff()
    vol30 = lr.rolling(30, min_periods=30).std() * np.sqrt(ANN)
    vol_pct = vol30.rolling(365, min_periods=120).rank(pct=True)
    hi365 = d.rolling(365, min_periods=1).max()
    sma20 = sma.shift(20)
    daily = pd.DataFrame({
        "close_d": d,
        "above_sma200": np.where(sma.isna(), np.nan, (d > sma).astype(float)),          # 1.0 / 0.0 / NaN（预热不足）
        "sma200_up": np.where(sma.isna() | sma20.isna(), np.nan, (sma > sma20).astype(float)),
        "mom20d": d / d.shift(20) - 1.0, "vol30d": vol30, "vol30d_pct": vol_pct, "dd365": d / hi365 - 1.0,
    }, index=d.index)
    avail = to_ns(daily.index) + NS_DAY
    j = np.searchsorted(avail, t_dec, side="right") - 1
    out = daily.iloc[np.clip(j, 0, None)].reset_index(drop=True)
    out.loc[j < 0, :] = np.nan
    out.insert(0, "daily_label", daily.index[np.clip(j, 0, None)])

    ev = crash_event_times(df4h)
    k = np.searchsorted(ev, t_dec, side="right") - 1
    hrs = np.full(len(bars), np.inf)
    ok = k >= 0
    hrs[ok] = (t_dec[ok] - ev[k[ok]]) / NS_HOUR
    out["hours_since_crash"] = hrs

    c = df1h["close"].to_numpy(dtype=np.float64)
    for lag, name in ((24, "ret24h"), (168, "ret168h")):
        r = np.full(len(bars), np.nan)
        okb = bars >= lag
        r[okb] = c[bars[okb]] / c[bars[okb] - lag] - 1.0
        out[name] = r
    return out


# ---------------------------------------------------------------------------
# 括号单撮合（与回测器 VAH 全平模式相同的规则）与随机入场零假设
# ---------------------------------------------------------------------------
@njit(cache=True)
def _bracket(o, h, l, c, entry_bar, sl_abs, tp_abs, end_bar, taker, maker, slip):
    m = entry_bar.shape[0]
    xbar = np.full(m, -1, np.int64)
    reason = np.zeros(m, np.int8)          # 1 SL, 2 TP, 0 未在 end_bar 之前了结（按 end_bar-1 收盘价 taker 离场）
    r_mult = np.full(m, np.nan)
    for j in range(m):
        b = entry_bar[j]
        efill = c[b] * (1.0 + slip)
        s = sl_abs[j]
        t = tp_abs[j]
        risk = efill - s
        px = np.nan
        fee_x = taker
        for i in range(b + 1, end_bar):
            if l[i] <= s:
                px = min(o[i], s) * (1.0 - slip)
                reason[j] = 1
                xbar[j] = i
                break
            if h[i] >= t:
                px = t * (1.0 - slip)
                fee_x = maker
                reason[j] = 2
                xbar[j] = i
                break
        if reason[j] == 0:
            xbar[j] = end_bar - 1
            px = c[end_bar - 1] * (1.0 - slip)
        pnl = px - efill - taker * efill - fee_x * px
        r_mult[j] = pnl / risk
    return xbar, reason, r_mult


def bracket_outcomes(df1h: pd.DataFrame, entry_bar, sl_abs, tp_abs, end_bar: int, bcfg: BacktestConfig,
                     allow_oos: bool = False):
    """在 entry_bar 收盘以 close*(1+slip) 入场、绝对价位 sl_abs / tp_abs 的括号单，于 end_bar 之前撮合。
    返回 (exit_bar, reason[1=SL, 2=TP, 0=未了结], R_ex_funding)。第 end_bar-1 根的收盘晚于 OOS_START 时须 allow_oos。"""
    _check_bars_extent(df1h, int(end_bar), allow_oos)
    o, h, l, c = (df1h[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close"))
    eb = np.asarray(entry_bar, dtype=np.int64)
    if len(eb) and (eb.min() < 0 or eb.max() >= end_bar - 1):
        raise ValueError("entry bars must lie in [0, end_bar - 1)")
    return _bracket(o, h, l, c, eb, np.asarray(sl_abs, dtype=np.float64), np.asarray(tp_abs, dtype=np.float64),
                    int(end_bar), float(bcfg.taker_fee), float(bcfg.maker_fee), float(bcfg.slippage))


def _sample_entry_bars(idx_ns: np.ndarray, entry_bars: np.ndarray, first: int, end_bar: int, n_rep: int,
                       rng: np.random.Generator, window_days: float | None, side: str,
                       exclude_bars: int | None) -> np.ndarray:
    """为每笔交易抽 n_rep 个随机入场 K 线（形状 n_rep × m）。可抽范围 = [first, end_bar - 1)（保证至少一根后续 K 线），
    window_days 给定时再限制为开盘时刻距信号 K 线开盘 ≤ window_days；side='before' / 'after' 只取信号 K 线之前 / 之后；
    exclude_bars = k 时剔除 |b - 信号 K 线| ≤ k 的 K 线（去掉与真实交易共享路径的入场）。"""
    m = len(entry_bars)
    hi_cap = end_bar - 1
    if window_days is None:
        lo = np.full(m, first, np.int64)
        hi = np.full(m, hi_cap, np.int64)
    else:
        w = np.int64(round(window_days * 86400 * 10**9))
        t_e = idx_ns[entry_bars]
        lo = np.maximum(np.searchsorted(idx_ns, t_e - w, side="left"), first).astype(np.int64)
        hi = np.minimum(np.searchsorted(idx_ns, t_e + w, side="right"), hi_cap).astype(np.int64)
    if side == "both" and exclude_bars is None:
        if (hi <= lo).any():
            raise ValueError("a trade's sampling window is empty")
        return lo[None, :] + np.floor(rng.random((n_rep, m)) * (hi - lo)[None, :]).astype(np.int64)
    k = 0 if exclude_bars is None else int(exclude_bars)
    left_lo, left_hi = lo, np.minimum(hi, entry_bars - k)              # [lo, e - k)
    right_lo, right_hi = np.maximum(lo, entry_bars + k + 1), hi        # [e + k + 1, hi)
    n_left = np.where(side in ("both", "before"), np.maximum(left_hi - left_lo, 0), 0)
    n_right = np.where(side in ("both", "after"), np.maximum(right_hi - right_lo, 0), 0)
    tot = n_left + n_right
    if (tot <= 0).any():
        raise ValueError("a trade's sampling pool is empty")
    u = np.floor(rng.random((n_rep, m)) * tot[None, :]).astype(np.int64)
    return np.where(u < n_left[None, :], left_lo[None, :] + u, right_lo[None, :] + (u - n_left[None, :]))


def random_entry_null(df1h: pd.DataFrame, tf: pd.DataFrame, bcfg: BacktestConfig, start: pd.Timestamp,
                      end: pd.Timestamp, n_rep: int = 2000, seed: int = 0, window_days: float | None = None,
                      side: str = "both", exclude_bars: int | None = None, allow_oos: bool = False) -> dict:
    """每笔真实交易保持 (d_sl, d_tp) 不变，入场换成 [start, end) 内随机 1H K 线的收盘（抽样范围见 _sample_entry_bars），
    撮合截止于 end 之前的最后一根 K 线。返回零分布与单侧 p 值。
    window_days 足够小且 side='both'、不剔除时只能抽到真实信号 K 线本身，零分布退化为真实结果（测试据此校验）。"""
    if side not in ("both", "before", "after"):
        raise ValueError("side must be 'both', 'before' or 'after'")
    _check_trades_extent(tf, allow_oos)
    if end > OOS_START and not allow_oos:
        raise OOSLockedError("random-entry nulls reaching beyond OOS_START require allow_oos=True")
    idx = df1h.index
    first = int(np.searchsorted(idx, start, side="left"))
    end_bar = int(np.searchsorted(idx, end, side="left"))          # 第一根 open ≥ end 的 K 线（不参与撮合）
    if end_bar - 1 <= first:
        raise ValueError("empty sampling window")
    rng = np.random.default_rng(seed)
    m = len(tf)
    d_sl = tf["d_sl"].to_numpy(dtype=np.float64)
    d_tp = tf["d_tp"].to_numpy(dtype=np.float64)
    bars = _sample_entry_bars(to_ns(idx), tf["entry_bar"].to_numpy(dtype=np.int64), first, end_bar, n_rep, rng,
                              window_days, side, exclude_bars)
    c = df1h["close"].to_numpy(dtype=np.float64)
    efill = c[bars.ravel()] * (1.0 + float(bcfg.slippage))
    sl_abs = efill * (1.0 - np.tile(d_sl, n_rep))
    tp_abs = efill * (1.0 + np.tile(d_tp, n_rep))
    _, reason, r = bracket_outcomes(df1h, bars.ravel(), sl_abs, tp_abs, end_bar, bcfg, allow_oos=allow_oos)
    wins = (r > 0).reshape(n_rep, m)
    rr = r.reshape(n_rep, m)
    w_null = wins.sum(axis=1)
    r_null = rr.mean(axis=1)
    w_act = int((tf["r_ex_funding"] > 0).sum())
    r_act = float(tf["r_ex_funding"].mean())
    return {
        "n_trades": m, "n_rep": n_rep, "seed": seed, "window_days": window_days, "side": side,
        "exclude_bars": exclude_bars,
        "actual_wins": w_act, "actual_mean_r_ex_funding": r_act,
        "null_wins_mean": float(w_null.mean()), "null_wins_p05": float(np.percentile(w_null, 5)),
        "null_wins_p95": float(np.percentile(w_null, 95)),
        "null_mean_r_mean": float(r_null.mean()), "null_mean_r_p05": float(np.percentile(r_null, 5)),
        "null_mean_r_p95": float(np.percentile(r_null, 95)),
        # 单侧 p 值（含实际值本身的 +1 修正）：零分布不高于实际值的比例 → 小 p 表示真实入场显著劣于随机
        "p_wins_le": float((1 + (w_null <= w_act).sum()) / (n_rep + 1)),
        "p_mean_r_le": float((1 + (r_null <= r_act + 1e-12).sum()) / (n_rep + 1)),   # 容差：只影响数值上的并列
        "unresolved_frac": float((reason == 0).mean()),
        "per_trade_null_win_rate": wins.mean(axis=0),
        "per_trade_null_mean_r": rr.mean(axis=0),          # 每笔交易在零假设下的期望 R（用于按组计算超额 R）
        "null_r_dist": r_null, "null_w_dist": w_null,
    }


def poisson_binomial_cdf(p: np.ndarray, k: int) -> float:
    """独立伯努利（成功概率 p_j）之和 ≤ k 的精确概率（动态规划）。"""
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for q in np.asarray(p, dtype=np.float64):
        dist[1:] = dist[1:] * (1.0 - q) + dist[:-1] * q
        dist[0] *= 1.0 - q
    return float(dist[: k + 1].sum())


# ---------------------------------------------------------------------------
# 回撤区间、分桶统计、夏普置信区间
# ---------------------------------------------------------------------------
def drawdown_episodes(eq: pd.Series, top: int = 5, allow_oos: bool = False) -> list[dict]:
    """按深度排序的回撤区间：峰值 → 谷底 → 收复（重新达到峰值；未收复则为 None）。"""
    _check_series_extent(eq, allow_oos)
    peak = eq.cummax()
    under = eq < peak
    eps = []
    i, n = 0, len(eq)
    vals = eq.to_numpy()
    pk = peak.to_numpy()
    while i < n:
        if not under.iloc[i]:
            i += 1
            continue
        s = i                                   # 回撤开始（峰值在 s-1）
        while i < n and under.iloc[i]:
            i += 1
        seg = vals[s:i] / pk[s:i] - 1.0
        tr = s + int(np.argmin(seg))
        eps.append({"peak": eq.index[s - 1] if s > 0 else eq.index[0], "trough": eq.index[tr],
                    "recovery": eq.index[i] if i < n else None, "depth": float(seg.min()),
                    "days_peak_to_trough": int((eq.index[tr] - (eq.index[s - 1] if s > 0 else eq.index[0])).days)})
    eps.sort(key=lambda e: e["depth"])
    return eps[:top]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(mid - half), float(mid + half)


def _boot_mean(x: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    """均值的自助法 95% 区间与标准误（样本量 1 时区间退化为该值、标准误为 NaN）。"""
    if len(x) < 2:
        return float(x.mean()), float(x.mean()), float("nan")
    b = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), float(b.std(ddof=1))


def bucket_stats(tf: pd.DataFrame, by, n_boot: int = 2000, seed: int = 0, allow_oos: bool = False) -> pd.DataFrame:
    """按 by（列名或与 tf 对齐的 Series）分组的交易统计。

    * 净胜率（pnl > 0，Wilson 95%）与止盈命中率（exit_reason == 'TP'，Wilson 95%）。与无漂移随机游走理论值
      rw_win_rate（组内 d_sl/(d_sl+d_tp) 的均值，即“先触及止盈”的概率，不含成本）可比的是止盈命中率。
    * 平均 R（含资金费，自助法 95%）；tf 含 r_ex_funding 时另给不含资金费的平均 R（mean_r_xf，自助法 95%）。
    * 净值口径：每笔占入场净值的平均盈亏 mean_ret（ret_on_base）与组内对数收益之和 sum_log_ret
      （复利、单仓位下 IS 期末/期初净值的对数 = 全部交易 log(1+ret_on_base) 之和）。
    * tf 含 null_r（每笔交易在随机入场零假设下的期望 R，不含资金费）时，给出组内零假设平均 R 与超额 R
      （r_ex_funding - null_r，自助法 95% 与标准误），此时 mean_r_xf - null_mean_r = excess_r 恒成立。
    """
    _check_trades_extent(tf, allow_oos)
    rng = np.random.default_rng(seed)
    key = tf[by] if isinstance(by, str) else by
    rows = []
    for g, sub in tf.groupby(key, observed=True, dropna=False):
        n = len(sub)
        k = int(sub["win"].sum())
        lo, hi = wilson(k, n)
        r = sub["r_multiple"].to_numpy()
        r_lo, r_hi, r_se = _boot_mean(r, rng, n_boot)
        row = {"bucket": g, "n": n, "wins": k, "win_rate": k / n, "win_lo": lo, "win_hi": hi,
               "rw_win_rate": float(sub["p_rw"].mean()), "mean_r": float(r.mean()), "mean_r_lo": r_lo,
               "mean_r_hi": r_hi, "sum_r": float(r.sum()), "sum_pnl": float(sub["pnl"].sum())}
        if "exit_reason" in sub.columns:
            k_tp = int((sub["exit_reason"] == "TP").sum())
            t_lo, t_hi = wilson(k_tp, n)
            row.update({"tp_hits": k_tp, "tp_rate": k_tp / n, "tp_lo": t_lo, "tp_hi": t_hi})
        if "ret_on_base" in sub.columns:
            ret = sub["ret_on_base"].to_numpy(dtype=np.float64)
            row.update({"mean_ret": float(ret.mean()), "sum_log_ret": float(np.log1p(ret).sum())})
        if "r_ex_funding" in sub.columns:
            xf = sub["r_ex_funding"].to_numpy(dtype=np.float64)
            x_lo, x_hi, x_se = _boot_mean(xf, rng, n_boot)
            row.update({"mean_r_xf": float(xf.mean()), "mean_r_xf_lo": x_lo, "mean_r_xf_hi": x_hi, "mean_r_xf_se": x_se})
        if "null_r" in sub.columns:
            ex = (sub["r_ex_funding"] - sub["null_r"]).to_numpy()
            e_lo, e_hi, e_se = _boot_mean(ex, rng, n_boot)
            row.update({"null_mean_r": float(sub["null_r"].mean()), "excess_r": float(ex.mean()),
                        "excess_r_lo": e_lo, "excess_r_hi": e_hi, "excess_r_se": e_se})
        rows.append(row)
    return pd.DataFrame(rows)


def _stationary_indices(n: int, n_boot: int, mean_block: float, seed: int) -> np.ndarray:
    """平稳自助法（Politis-Romano）的重抽样下标：块长服从均值 mean_block 的几何分布，环形拼接。"""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n))
    new_block = rng.random((n_boot, n)) < 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _ann_sharpe_rows(xb: np.ndarray) -> np.ndarray:
    sd = xb.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return xb.mean(axis=1) / sd * np.sqrt(ANN)


def stationary_bootstrap_sharpe(r: pd.Series, n_boot: int = 5000, mean_block: float = 10.0, seed: int = 0,
                                allow_oos: bool = False) -> dict:
    """日度收益的平稳自助法（几何分布块长，均值 mean_block 天）年化夏普置信区间（百分位法）。"""
    _check_series_extent(r, allow_oos)
    x = r.to_numpy(dtype=np.float64)
    n = len(x)
    s = _ann_sharpe_rows(x[_stationary_indices(n, n_boot, mean_block, seed)])
    s = s[np.isfinite(s)]
    point = float(x.mean() / x.std(ddof=1) * np.sqrt(ANN))
    return {"sharpe": point, "lo": float(np.percentile(s, 2.5)), "hi": float(np.percentile(s, 97.5)),
            "p_gt0": float((s > 0).mean()), "n_days": n, "mean_block": mean_block, "n_boot": n_boot}


def paired_sharpe_diff(r_new: pd.Series, r_ref: pd.Series, n_boot: int = 5000, mean_block: float = 10.0, seed: int = 0,
                       allow_oos: bool = False) -> dict:
    """同一时段两条日度收益序列的夏普差 Sharpe(new) - Sharpe(ref)：两者用同一组平稳自助法下标重抽样，
    保留同日相关性（同一批信号的两个变体高度相关，单独的置信区间会严重高估差值的不确定性）。"""
    _check_series_extent(r_new, allow_oos)
    _check_series_extent(r_ref, allow_oos)
    if not r_new.index.equals(r_ref.index):
        raise ValueError("paired series must share the same daily index")
    a, b = r_new.to_numpy(dtype=np.float64), r_ref.to_numpy(dtype=np.float64)
    idx = _stationary_indices(len(a), n_boot, mean_block, seed)
    d = _ann_sharpe_rows(a[idx]) - _ann_sharpe_rows(b[idx])
    d = d[np.isfinite(d)]
    point = float((a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1)) * np.sqrt(ANN))
    return {"diff": point, "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p_gt0": float((d > 0).mean()), "n_days": len(a), "mean_block": mean_block, "n_boot": n_boot}


def segment_trades(tf: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, allow_oos: bool = False) -> pd.DataFrame:
    """按入场时刻 ∈ [start, end) 取交易（与 metrics.segment_metrics 的归属一致）。"""
    _check_trades_extent(tf, allow_oos)
    return tf[(tf["entry_time"] >= start) & (tf["entry_time"] < end)].copy()


def segment_daily_returns(res: BacktestResult, start: pd.Timestamp, end: pd.Timestamp, allow_oos: bool = False) -> pd.Series:
    """与 metrics.segment_metrics 完全相同的分段日度收益（首个收益日 = start，含不复利配置的重新定基）。"""
    d_eq = daily_equity(res, allow_oos)
    seg = d_eq[(d_eq.index >= start - DAY) & (d_eq.index <= end - DAY)]
    if len(seg) < 2 or seg.index[0] != start - DAY or seg.index[-1] != end - DAY:
        raise ValueError(f"segment [{start}, {end}) not covered by backtest equity")
    if not res.config.compounding:
        seg = seg + (res.initial_capital - seg.iloc[0])
    return daily_returns(seg, allow_oos)


def buy_and_hold(df1h: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, allow_oos: bool = False) -> dict:
    """持有 1 单位永续合约价格（不计费用、滑点与资金费）的分段日度指标，日度口径与 segment_metrics 相同：
    日 D 的净值 = D+1 00:00 的价格（即当日最后一根 1H 收盘）；起点锚 = start 00:00 的价格（数据首根则取其开盘价）。"""
    px = pd.Series(df1h["close"].to_numpy(dtype=np.float64), index=df1h.index + pd.Timedelta(hours=1))
    px = pd.concat([pd.Series([float(df1h["open"].iloc[0])], index=[df1h.index[0]]), px])
    _check_series_extent(px[px.index <= end], allow_oos)
    lab = (px.index - pd.Timedelta(1, "ns")).floor("D")
    d = px.groupby(lab).last()
    d.index = pd.DatetimeIndex(d.index)
    if df1h.index[0] == start:                              # 数据首日：起点锚 = 首根开盘价，标注在前一天
        d = pd.concat([pd.Series([float(df1h["open"].iloc[0])], index=[start - DAY]), d[d.index >= start]])
    seg = d[(d.index >= start - DAY) & (d.index <= end - DAY)]
    if len(seg) < 2 or seg.index[0] != start - DAY or seg.index[-1] != end - DAY:
        raise ValueError(f"segment [{start}, {end}) not covered by price data")
    r = daily_returns(seg, allow_oos)
    days = len(r)
    mdd = max_drawdown(seg, allow_oos)
    cagr = (seg.iloc[-1] / seg.iloc[0]) ** (ANN / days) - 1.0
    return {"sharpe": sharpe(r, allow_oos), "sortino": sortino(r, allow_oos), "cagr": float(cagr),
            "total_return": float(seg.iloc[-1] / seg.iloc[0] - 1.0), "max_dd": mdd,
            "calmar": float(cagr / abs(mdd)) if mdd < 0 else float("nan"), "ann_vol": float(r.std(ddof=1) * np.sqrt(ANN)),
            "days": days}

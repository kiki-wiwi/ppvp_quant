"""
backtester.py — 事件驱动回测引擎（1H 粒度，只做多，单品种单仓位）。

执行与成本约定
--------------
* 入场：信号 K 线收盘后以收盘价 * (1 + 滑点) 市价成交，收 taker 费；止盈止损检查从入场后的下一根 1H 开始。
* 止损：stop-market，触发条件 low ≤ SL，成交价 = min(open, SL) * (1 - 滑点)（跳空低开按开盘价成交），taker 费。
* 止盈：VAH 限价单，触发条件 high ≥ TP，成交价 = TP * (1 - 滑点)，maker 费（跳空高开也只按 TP 计，保守；
  滑点用于刻画排队/部分成交风险）。
* K 线内悲观撮合：同一根 K 线同时满足 low ≤ SL 与 high ≥ TP 时，一律按止损出局。
  POC 分批模式下，若同一根 K 线先触及 POC 又回落到保本价，同样悲观地按保本价离场；
  partial_frac = 1.0 时全部仓位在 POC 平仓，离场原因记为 'POC'。
* 持仓期放量破位离场（exit_on_breakdown）：参照位冻结为“入场时箱体”的支撑带下沿 VAL_entry - k1*ATR_i，
  条件 = 阴线、收盘 < 参照位、实体 ≥ break_body_atr*ATR、量 ≥ break_vol_mult*均量（ATR/均量为第 i 根开盘时已知值），
  按收盘价市价离场。持仓期间新确认的箱体不改变该参照位。
* 资金费率（venue='perp' 且 include_funding=True）：逐个结算时点 T（Binance 为 UTC 00/08/16）结算，
  只在持仓实际跨越 T 时收付，金额 = 持仓数量 * T 时刻价格 * 当期真实历史费率（多头费率为正时支付、为负时收取）。
  T 映射到第一根 open_time ≥ T 的 1H K 线（正常即开盘恰为 T 的那根；若该根因维护缺失则顺延到下一根，
  价格取该根开盘价），条件为 入场时刻 < T。收盘市价离场且收盘时刻恰为 T 时，按收盘价承担该次资金费。
  成交被视为发生在 K 线收盘瞬间之后：收盘入场不承担该时刻的资金费。
  开启资金费时必须传入资金费表，且 (第一根开盘, 最后收盘) 内的每个结算网格点都必须存在、窗口内每个时间戳都必须
  落在网格上且费率非 NaN，否则直接报错（不允许静默按 0 计，见 check_funding_coverage）；恰在最后收盘时刻的结算
  若缺失，只影响在最后一根收盘离场的时间止损/破位离场，这类交易被标记为 funding_complete=False。
  交易表 n_funding 记录每笔交易实际结算的次数；SL/TP/BE/POC 离场的资金费窗口止于 exit_bar_open，
  TIME/BREAKDOWN 止于 exit_time（离场 K 线收盘）。
* venue='spot'：现货等效模式 = 永续价格触发与成交 + 现货费率 + 无资金费 + 杠杆 ≤ 1.0，
  且名义仓位 ≤ 净值 / (1 + taker)，保证买入后现金不为负（手续费不融资）。
* 仓位：fixed（净值 * 固定杠杆）/ risk（以损定仓：净值 * 风险比例 / 止损距离）/ inv_vol（目标波动率缩放），
  均受 leverage_cap * 净值 硬上限约束；该上限只在入场时按当时净值执行，持仓期间盯市杠杆会随价格下跌而升高。
  上限截断了目标名义的交易标记 capped=True（不复利配置下，这意味着当期净值而非初始资金决定了仓位）。
  ret_on_base = pnl / 仓位基数（复利 = 入场时净值，不复利 = 初始资金）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import BacktestConfig
from numba_compat import njit
from ppvp_engine import to_ns

REASONS = {0: "OPEN", 1: "SL", 2: "TP", 3: "BE", 4: "TIME", 5: "BREAKDOWN", 6: "POC"}
NS_PER_HOUR = np.int64(3600 * 10**9)


@njit(cache=True)
def _simulate(o, h, l, c, open_ns, close_ns, sig, sl_arr, tp_arr, poc_arr, val_arr, bodyvol, k1atr,
              fund_open, fund_t, fund_close, ann_vol,
              init_cap, taker, maker, slip, funding_on, spot_cash,
              sizing, fixed_lev, risk_pct, target_vol, lev_cap, compounding,
              tp_mode, partial_frac, move_be, time_stop, exit_on_brk):
    n = o.shape[0]
    equity = np.empty(n)
    pos = np.zeros(n)
    m = 1
    for i in range(n):
        if sig[i]:
            m += 1
    t_entry_bar = np.full(m, -1, np.int64)
    t_exit_bar = np.full(m, -1, np.int64)
    t_entry_px = np.full(m, np.nan)
    t_exit_px = np.full(m, np.nan)
    t_qty = np.full(m, np.nan)
    t_notional = np.full(m, np.nan)
    t_lev = np.full(m, np.nan)
    t_eq = np.full(m, np.nan)
    t_sl = np.full(m, np.nan)
    t_tp = np.full(m, np.nan)
    t_poc = np.full(m, np.nan)
    t_fees = np.zeros(m)
    t_fund = np.zeros(m)
    t_nfund = np.zeros(m, np.int64)
    t_gross = np.zeros(m)
    t_pnl = np.full(m, np.nan)
    t_reason = np.zeros(m, np.int8)
    t_partial_bar = np.full(m, -1, np.int64)
    t_partial_px = np.full(m, np.nan)
    t_min_low = np.full(m, np.nan)
    t_max_high = np.full(m, np.nan)
    t_capped = np.zeros(m, np.bool_)
    lv_stop = np.full(n, np.nan)          # 逐根：本根开始时生效的止损位（含此前的保本上移；入场 K 线为 NaN）
    lv_brk = np.full(n, np.nan)           # 逐根：本根生效的破位离场参照位 VAL_entry - k1*ATR_i（仅 exit_on_breakdown）
    nt = 0

    wallet = init_cap
    qty = 0.0
    qty0 = 0.0
    entry_px = 0.0
    entry_bar = -1
    entry_t = np.int64(0)
    sl = np.nan
    tp = np.nan
    poc = np.nan
    val_e = np.nan
    partial_done = False
    gross = 0.0
    fees = 0.0
    fund = 0.0
    nfund = 0
    exit_val = 0.0
    min_low = np.inf
    max_high = -np.inf
    k = -1
    for i in range(n):
        if qty > 0.0:
            # --- 本根生效的离场价位：只计算一次并逐根记录，离场判定只读取记录值（闸门逐位比较这些价位）---
            lv_stop[i] = sl
            brk_lvl = val_e - k1atr[i]
            if exit_on_brk:
                lv_brk[i] = brk_lvl
            # --- 资金费：结算时点 T 映射到第一根 open ≥ T 的 K 线，且入场早于 T ---
            fr = fund_open[i]
            if funding_on and fr == fr and entry_t < fund_t[i]:
                f = qty * o[i] * fr
                wallet -= f
                fund += f
                nfund += 1
            if l[i] < min_low:
                min_low = l[i]
            if h[i] > max_high:
                max_high = h[i]
            reason = 0
            # --- 止损优先（悲观撮合）---
            if l[i] <= lv_stop[i]:
                px = min(o[i], lv_stop[i]) * (1.0 - slip)
                fee = qty * px * taker
                g = qty * (px - entry_px)
                wallet += g - fee
                gross += g
                fees += fee
                exit_val += qty * px
                qty = 0.0
                reason = 3 if (partial_done and move_be) else 1
            else:
                # --- POC 分批止盈 ---
                if tp_mode == 1 and (not partial_done) and poc > entry_px and poc < tp and h[i] >= poc:
                    q1 = qty * partial_frac
                    px = poc * (1.0 - slip)
                    fee = q1 * px * maker
                    g = q1 * (px - entry_px)
                    wallet += g - fee
                    gross += g
                    fees += fee
                    exit_val += q1 * px
                    qty -= q1
                    partial_done = True
                    t_partial_bar[k] = i
                    t_partial_px[k] = px
                    if qty <= 0.0:                        # partial_frac = 1：全部在 POC 平仓
                        qty = 0.0
                        reason = 6
                    elif move_be:
                        sl = entry_px
                        if l[i] <= sl:                    # 同根 K 线回落至保本价：悲观地按保本离场
                            px = sl * (1.0 - slip)
                            fee = qty * px * taker
                            g = qty * (px - entry_px)
                            wallet += g - fee
                            gross += g
                            fees += fee
                            exit_val += qty * px
                            qty = 0.0
                            reason = 3
                # --- VAH 止盈 ---
                if qty > 0.0 and h[i] >= tp:
                    px = tp * (1.0 - slip)
                    fee = qty * px * maker
                    g = qty * (px - entry_px)
                    wallet += g - fee
                    gross += g
                    fees += fee
                    exit_val += qty * px
                    qty = 0.0
                    reason = 2
            # --- 收盘类离场：时间止损 / 放量跌穿入场箱体支撑带 ---
            if qty > 0.0:
                ts_hit = time_stop > 0 and (i - entry_bar) >= time_stop
                bd_hit = exit_on_brk and bodyvol[i] and c[i] < lv_brk[i]
                if ts_hit or bd_hit:
                    fr2 = fund_close[i]
                    if funding_on and fr2 == fr2:
                        f = qty * c[i] * fr2
                        wallet -= f
                        fund += f
                        nfund += 1
                    px = c[i] * (1.0 - slip)
                    fee = qty * px * taker
                    g = qty * (px - entry_px)
                    wallet += g - fee
                    gross += g
                    fees += fee
                    exit_val += qty * px
                    qty = 0.0
                    reason = 4 if ts_hit else 5
            if qty == 0.0:
                t_exit_bar[k] = i
                t_exit_px[k] = exit_val / qty0
                t_fees[k] = fees
                t_fund[k] = fund
                t_nfund[k] = nfund
                t_gross[k] = gross
                t_pnl[k] = gross - fees - fund
                t_reason[k] = reason
                t_min_low[k] = min_low
                t_max_high[k] = max_high
        equity[i] = wallet + qty * (c[i] - entry_px) if qty > 0.0 else wallet

        # --- 入场（空仓且本根收盘出现信号）---
        if qty == 0.0 and sig[i]:
            eq = equity[i]
            base = eq if compounding else init_cap
            efill = c[i] * (1.0 + slip)
            notional = 0.0
            if sizing == 0:
                notional = base * fixed_lev
            elif sizing == 1:
                dist = (efill - sl_arr[i]) / efill
                if dist > 0.0:
                    notional = base * risk_pct / dist
            else:
                av = ann_vol[i]
                if av == av and av > 0.0:
                    notional = base * target_vol / av
            cap = lev_cap * eq
            if spot_cash:
                cash_cap = eq / (1.0 + taker)
                if cash_cap < cap:
                    cap = cash_cap
            capped = notional > cap
            if capped:
                notional = cap
            if notional > 0.0 and eq > 0.0 and efill > sl_arr[i]:
                k = nt
                nt += 1
                qty = notional / efill
                qty0 = qty
                fee = notional * taker
                wallet -= fee
                entry_px = efill
                entry_bar = i
                entry_t = close_ns[i]
                sl = sl_arr[i]
                tp = tp_arr[i]
                poc = poc_arr[i]
                val_e = val_arr[i]
                partial_done = False
                gross = 0.0
                fees = fee
                fund = 0.0
                nfund = 0
                exit_val = 0.0
                min_low = np.inf
                max_high = -np.inf
                t_entry_bar[k] = i
                t_entry_px[k] = efill
                t_qty[k] = qty
                t_notional[k] = notional
                t_lev[k] = notional / eq
                t_eq[k] = eq
                t_sl[k] = sl
                t_tp[k] = tp
                t_poc[k] = poc
                t_capped[k] = capped
                equity[i] = wallet + qty * (c[i] - entry_px)
        pos[i] = qty
    # 期末仍持仓：按最后收盘价盯市，记录为 OPEN
    if qty > 0.0 and k >= 0:
        t_fees[k] = fees
        t_fund[k] = fund
        t_nfund[k] = nfund
        t_gross[k] = gross + qty * (c[n - 1] - entry_px)
        t_pnl[k] = t_gross[k] - fees - fund
        t_reason[k] = 0
        t_min_low[k] = min_low
        t_max_high[k] = max_high
    return (equity, pos, nt, t_entry_bar, t_exit_bar, t_entry_px, t_exit_px, t_qty, t_notional, t_lev, t_eq,
            t_sl, t_tp, t_poc, t_fees, t_fund, t_nfund, t_gross, t_pnl, t_reason, t_partial_bar, t_partial_px,
            t_min_low, t_max_high, t_capped, lv_stop, lv_brk)


@dataclass(repr=False)
class BacktestResult:
    equity: pd.Series          # 索引 = 1H 收盘时刻 close_time，值 = 该时刻盯市净值
    position: pd.Series        # 索引 = close_time，值 = 持仓数量
    trades: pd.DataFrame
    config: BacktestConfig
    initial_capital: float
    start_time: pd.Timestamp   # 第一根 1H 开盘时刻（净值起点）
    funding_end_missing: bool = False   # 资金费表缺少恰在最后收盘时刻的结算（见 check_funding_coverage）
    levels: pd.DataFrame | None = None  # 逐根持仓离场价位（stop / brk_level），供防未来函数闸门逐位比较

    def __repr__(self) -> str:
        # 只显示规模与时间范围、不显示任何数值：测试失败时 pytest 会渲染局部变量/参数（-l、-vv），不得借此输出 OOS 净值
        span = f"{self.equity.index[0]} .. {self.equity.index[-1]}" if len(self.equity) else "empty"
        return f"BacktestResult(bars={len(self.equity)}, span={span}, trades={len(self.trades)})"


def funding_arrays(df1h: pd.DataFrame, funding: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把结算时点 T 的真实费率映射到 1H K 线。

    返回 (fund_open, fund_t, fund_close)：
    * fund_open[i] / fund_t[i]：映射到第 i 根的结算费率与结算时刻 T（第 i 根 = 第一根 open_time ≥ T 的 K 线）；
    * fund_close[i]：close_time[i] 恰为结算时刻时的费率（供收盘市价离场使用）。
    只考虑 open_time[0] < T ≤ 最后收盘 的结算点；两个结算点落到同一根 K 线（数据缺口 > 结算间隔）时报错。
    """
    n = len(df1h)
    fo, ft, fc = np.full(n, np.nan), np.zeros(n, dtype=np.int64), np.full(n, np.nan)
    if funding is None or len(funding) == 0 or n == 0:
        return fo, ft, fc
    t_ns = to_ns(funding.index)
    fr = funding["funding_rate"].to_numpy(dtype=np.float64)
    o_ns = to_ns(df1h.index)
    c_ns = o_ns + NS_PER_HOUR
    win = (t_ns > o_ns[0]) & (t_ns <= c_ns[-1])
    t_ns, fr = t_ns[win], fr[win]
    j = np.searchsorted(o_ns, t_ns, side="left")
    ok = j < n
    if np.unique(j[ok]).size != int(ok.sum()):
        raise ValueError("two funding settlements map to the same 1H bar (data gap longer than the settlement interval)")
    fo[j[ok]] = fr[ok]
    ft[j[ok]] = t_ns[ok]
    kk = np.searchsorted(c_ns, t_ns, side="left")
    kc = np.minimum(kk, n - 1)
    hit = (kk < n) & (c_ns[kc] == t_ns)
    fc[kk[hit]] = fr[hit]
    return fo, ft, fc


def check_funding_coverage(df1h: pd.DataFrame, funding: pd.DataFrame) -> bool:
    """资金费表在回测窗口 (第一根开盘, 最后收盘] 内必须完整、规整，否则报错；返回“恰在最后收盘时刻的网格点是否缺失”。

    * 结算网格 = 自 UTC 纪元起每 funding_interval_hours 小时（Binance BTC/ETH 为 8h → 00/08/16）。
      该列必须存在，窗口前后 1 天内的取值必须有限、为正整数且唯一（不支持窗口内改变结算间隔）；
      窗口前后 1 天内没有任何结算记录时无法确定网格，直接报错。
    * 窗口内的每个结算时间戳都必须落在网格上、费率不得为 NaN（funding_arrays 会对窗口内每个时间戳收付，
      不在网格上的时间戳或 NaN 费率都会造成静默错计）；重复时间戳报错。
    * 窗口内除最后收盘时刻以外的每个网格点都必须存在。恰在最后收盘时刻的结算只会被“在最后一根收盘离场”的
      时间止损/破位离场用到；若缺失，回测不报错，而是在交易表中把这类交易标记为 funding_complete=False。
    * 网格用整数纳秒运算生成（np.arange 以 float64 估算长度，窗口超过约 104 天时会丢掉终点）。
    """
    if funding.index.has_duplicates:
        raise ValueError("funding table has duplicate settlement timestamps")
    if "funding_interval_hours" not in funding.columns:
        raise ValueError("funding table lacks the funding_interval_hours column (needed to define the settlement grid)")
    o0 = df1h.index[0]
    cl = df1h.index[-1] + pd.Timedelta(hours=1)
    near = funding.loc[(funding.index > o0 - pd.Timedelta(days=1)) & (funding.index <= cl + pd.Timedelta(days=1)),
                       "funding_interval_hours"].to_numpy(dtype=np.float64)
    if len(near) == 0:
        raise ValueError(f"funding table has no settlement within one day of the backtest window [{o0}, {cl}]")
    if not np.isfinite(near).all():
        raise ValueError("funding_interval_hours is missing near the backtest window")
    vals = np.unique(near)
    if len(vals) > 1:
        raise ValueError("funding interval changes inside the backtest window (unsupported)")
    if not (vals[0] > 0 and float(vals[0]).is_integer()):
        raise ValueError(f"funding_interval_hours must be a positive integer, got {vals[0]}")
    step = int(vals[0]) * int(NS_PER_HOUR)
    o0n, cln = int(to_ns(pd.DatetimeIndex([o0]))[0]), int(to_ns(pd.DatetimeIndex([cl]))[0])
    have = to_ns(funding.index)
    win = (have > o0n) & (have <= cln)
    off = have[win] % step != 0
    if off.any():
        bad = pd.Timestamp(int(have[win][off][0]), unit="ns", tz="UTC")
        raise ValueError(f"funding settlement at {bad} is off the {int(vals[0])}h settlement grid")
    if np.isnan(funding["funding_rate"].to_numpy(dtype=np.float64)[win]).any():
        raise ValueError("funding table has NaN rates inside the backtest window")
    first = (o0n // step + 1) * step
    n_pts = max(0, (cln - first) // step + 1)
    if n_pts == 0:
        return False
    grid = np.int64(first) + np.int64(step) * np.arange(n_pts, dtype=np.int64)    # 网格点 ∈ (o0, cl]
    present = np.isin(grid, have[win])
    end_on_grid = int(grid[-1]) == cln
    inner = present[:-1] if end_on_grid else present
    if not inner.all():
        miss = pd.Timestamp(int(grid[:len(inner)][~inner][0]), unit="ns", tz="UTC")
        raise ValueError(f"funding table misses the settlement at {miss} inside the backtest window [{o0}, {cl}]")
    return bool(end_on_grid and not present[-1])


def annualized_vol(df1h: pd.DataFrame, lookback: int) -> np.ndarray:
    """第 i 根收盘时已知的 1H 对数收益滚动年化波动率。"""
    r = np.log(df1h["close"]).diff()
    return (r.rolling(lookback, min_periods=lookback).std() * np.sqrt(24 * 365)).to_numpy()


TRADE_SIGNAL_COLS = ("pattern", "box_id", "VAL", "POC", "VAH", "P_low", "t1_low", "t1_bar", "t2_low", "t2_bar",
                     "t2_start", "rr", "atr", "zone_lo", "zone_hi")


def run_backtest(df1h: pd.DataFrame, signals: pd.DataFrame, cfg: BacktestConfig,
                 funding: pd.DataFrame | None = None, use_jit: bool = True) -> BacktestResult:
    if not signals.index.equals(df1h.index):
        raise ValueError("signals must be aligned with df1h")
    n = len(df1h)
    for col in ("signal", "sl", "tp", "poc_tp"):
        if col not in signals.columns:
            raise ValueError(f"signals lacks required column '{col}'")
    if cfg.exit_on_breakdown:
        for col in ("VAL", "brk_bodyvol", "k1_atr"):
            if col not in signals.columns:
                raise ValueError(f"exit_on_breakdown=True requires signals column '{col}'")
        if (signals["signal"].to_numpy(dtype=bool) & signals["VAL"].isna().to_numpy()).any():
            raise ValueError("exit_on_breakdown=True requires a finite VAL on every signal bar")
    end_missing = False
    if cfg.funding_on:
        if funding is None:
            raise ValueError("include_funding=True on venue='perp' requires the funding table "
                             "(pass funding=..., or set include_funding=False)")
        end_missing = check_funding_coverage(df1h, funding)

    def col(name, dtype, fill):
        return signals[name].to_numpy(dtype=dtype) if name in signals.columns else np.full(n, fill, dtype=dtype)

    o, h, l, c = (df1h[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close"))
    open_ns = to_ns(df1h.index)
    close_ns = open_ns + NS_PER_HOUR
    fo, ft, fc = funding_arrays(df1h, funding if cfg.funding_on else None)
    av = annualized_vol(df1h, cfg.vol_lookback_bars) if cfg.sizing == "inv_vol" else np.full(n, np.nan)
    fn = _simulate if use_jit else _simulate.py_func
    out = fn(o, h, l, c, open_ns, close_ns,
             col("signal", np.bool_, False), col("sl", np.float64, np.nan), col("tp", np.float64, np.nan),
             col("poc_tp", np.float64, np.nan), col("VAL", np.float64, np.nan), col("brk_bodyvol", np.bool_, False),
             col("k1_atr", np.float64, np.nan), fo, ft, fc, av,
             float(cfg.initial_capital), float(cfg.taker_fee), float(cfg.maker_fee), float(cfg.slippage),
             bool(cfg.funding_on), bool(cfg.venue == "spot"),
             {"fixed": 0, "risk": 1, "inv_vol": 2}[cfg.sizing], float(cfg.fixed_leverage),
             float(cfg.risk_per_trade), float(cfg.target_vol), float(cfg.leverage_cap), bool(cfg.compounding),
             {"vah": 0, "poc_partial": 1}[cfg.tp_mode], float(cfg.partial_frac), bool(cfg.move_to_breakeven),
             int(cfg.time_stop_bars or -1), bool(cfg.exit_on_breakdown))
    (equity, pos, nt, eb, xb, epx, xpx, qty, notional, lev, eq, sl, tp, poc, fees, fund, nfund, gross, pnl, reason,
     pbar, ppx, mlow, mhigh, capped, lv_stop, lv_brk) = out
    eb, xb = eb[:nt], xb[:nt]

    def ts(ns_arr):
        return pd.DatetimeIndex(np.asarray(ns_arr, dtype=np.int64).view("datetime64[ns]")).tz_localize("UTC")

    nat = np.iinfo(np.int64).min
    closed = xb >= 0
    xb0 = np.where(closed, xb, 0)
    trades = pd.DataFrame({
        "entry_bar": eb, "exit_bar": xb,
        "entry_time": ts(close_ns[eb]),                                          # 信号 K 线收盘 = 入场时刻
        "exit_bar_open": ts(np.where(closed, open_ns[xb0], nat)),
        "exit_time": ts(np.where(closed, close_ns[xb0], nat)),                   # 离场所在 K 线的收盘时刻（保守记账）
        "entry_px": epx[:nt], "exit_px": xpx[:nt], "qty": qty[:nt], "notional": notional[:nt],
        "leverage": lev[:nt], "equity_at_entry": eq[:nt], "sl": sl[:nt], "tp": tp[:nt], "poc": poc[:nt],
        "fees": fees[:nt], "funding": fund[:nt], "n_funding": nfund[:nt], "gross_pnl": gross[:nt], "pnl": pnl[:nt],
        "exit_reason": pd.Series([REASONS[int(r)] for r in reason[:nt]], dtype="str"),   # 显式 dtype：空表一致
        "partial_bar": pbar[:nt], "partial_px": ppx[:nt], "min_low": mlow[:nt], "max_high": mhigh[:nt],
        "capped": capped[:nt],                                                   # 入场名义被净值上限截断
    })
    final_close_exit = (xb == n - 1) & np.isin(reason[:nt], (4, 5))
    trades["funding_complete"] = ~(final_close_exit & end_missing)
    # 派生列：空表也保持完整 schema（截断测试可逐位比较）
    risk = trades["qty"] * (trades["entry_px"] - trades["sl"])
    trades["r_multiple"] = trades["pnl"] / risk
    trades["ret_on_equity"] = trades["pnl"] / trades["equity_at_entry"]
    trades["ret_on_base"] = trades["pnl"] / (trades["equity_at_entry"] if cfg.compounding else float(cfg.initial_capital))
    trades["hold_bars"] = np.where(closed, xb - eb, n - 1 - eb).astype(np.int64)
    trades["mae_pct"] = trades["min_low"] / trades["entry_px"] - 1
    trades["mfe_pct"] = trades["max_high"] / trades["entry_px"] - 1
    for name in TRADE_SIGNAL_COLS:
        if name in signals.columns:
            trades[name] = signals[name].to_numpy()[eb]
    idx_close = df1h.index + pd.Timedelta(hours=1)
    return BacktestResult(equity=pd.Series(equity, index=idx_close, name="equity"),
                          position=pd.Series(pos, index=idx_close, name="qty"),
                          trades=trades, config=cfg, initial_capital=float(cfg.initial_capital),
                          start_time=df1h.index[0], funding_end_missing=bool(end_missing),
                          levels=pd.DataFrame({"stop": lv_stop, "brk_level": lv_brk}, index=idx_close))

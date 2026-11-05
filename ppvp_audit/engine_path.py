"""
ppvp_audit/engine_path.py — 通用的“子 K 线路径”撮合器。本轮按用户第 9 节补充约束，只用本地已有的 1H 数据：
子 K 线就是 1H K 线本身（每小时一根），K 线内的先后一律按事先写定的保守（最坏）规则处理，并单独披露歧义笔数。

用途：C1 的限价回撤单（1H 引擎 backtester.py 只支持市价入场），以及对 1H 引擎的逐位一致性核对。
与 backtester.py 相同的约定（市价入场、止损优先、止盈 maker、资金费、盯市净值、1 倍名义复利、单仓位）：
* 市价入场：信号 K 线收盘按收盘价 × (1 + 滑点) 成交，taker；止损 / 止盈从下一根起检查。
* 止损：触发 low ≤ SL，成交价 min(该根开盘, SL) × (1 − 滑点)，taker；止盈：high ≥ TP，成交价 TP × (1 − 滑点)，maker。
* 同一根同时满足 SL 与 TP（或 POC）时一律先止损；POC 分批后若同一根回落到保本价，按保本离场（最坏路径）。
* 资金费：结算时刻 T 映射到第一根开盘 ≥ T 的子 K 线；入场时刻 < T 且该根开始时仍持仓即收付，名义 = 数量 × 该根开盘价。
* 净值：每个 1H 收盘按 1H 收盘价盯市；仓位 = 入场时净值 × 1 倍名义，受 leverage_cap × 净值上限约束；
  同一时间只有一个仓位或一个挂单。
限价单（C1）：在信号 K 线收盘挂出买入限价 L，自下一根起有效 valid_bars 根。L ≥ 信号收盘时挂单即可成交，
按市价单处理（收盘价 × (1 + 滑点)，taker）。否则成交规则 'through'（穿价）：某根 low ≤ L − tick；'touch'（触价）：
low ≤ L。成交价 L × (1 + 滑点)，maker 费。最坏路径：成交那根若 low ≤ SL 则视为成交后再止损（同根离场）；
成交那根的止盈不计。到期未成交即取消；挂单或持仓期间出现的新信号记为 skipped。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from numba_compat import njit

REASONS_PATH = {0: "OPEN", 1: "SL", 2: "TP", 3: "BE", 6: "POC"}


@njit(cache=True)
def _sim(mo, mh, ml, mhour, h_close, n_hours, sig, sl_h, tp_h, poc_h, lim_h, tick_h, valid_min,
         fund_min, fund_rate, fund_t, open_ns, bar_ns, init_cap, taker, maker, slip, lev, lev_cap, tp_mode, partial_frac,
         move_be, limit_mode, through):
    M = mo.shape[0]
    eq_h = np.empty(n_hours)
    pos_h = np.zeros(n_hours)
    cap = n_hours + 1
    t_parent = np.full(cap, -1, np.int64)
    t_entry_min = np.full(cap, -1, np.int64)
    t_exit_min = np.full(cap, -1, np.int64)
    t_entry_px = np.full(cap, np.nan)
    t_exit_px = np.full(cap, np.nan)
    t_qty = np.full(cap, np.nan)
    t_notional = np.full(cap, np.nan)
    t_eq = np.full(cap, np.nan)
    t_sl = np.full(cap, np.nan)
    t_tp = np.full(cap, np.nan)
    t_poc = np.full(cap, np.nan)
    t_fees = np.zeros(cap)
    t_fund = np.zeros(cap)
    t_gross = np.zeros(cap)
    t_pnl = np.full(cap, np.nan)
    t_reason = np.zeros(cap, np.int8)
    t_partial_min = np.full(cap, -1, np.int64)
    o_status = np.zeros(n_hours, np.int8)      # 0 无信号，1 成交，2 到期未成交，3 因持仓或挂单被跳过，4 限价挂出即成交
    o_fill_min = np.full(n_hours, -1, np.int64)
    nt = 0
    wallet = init_cap
    qty = 0.0
    qty0 = 0.0
    entry_px = 0.0
    entry_min = -1
    entry_t = np.int64(0)    # 入场时刻（ns）：资金费只在 入场时刻 < 结算时刻 T 时收付（与 backtester.py 相同的时间规则）
    sl = np.nan
    tp = np.nan
    poc = np.nan
    partial_done = False
    gross = 0.0
    fees = 0.0
    fund = 0.0
    exit_val = 0.0
    k = -1
    pend = False
    pend_parent = -1
    pend_lim = 0.0
    pend_tick = 0.0
    pend_start = 0
    pend_end = 0
    fi = 0
    nf = fund_min.shape[0]
    for m in range(M):
        h = mhour[m]
        # ---- 资金费：结算分钟开始时仍持仓、且入场早于该分钟 ----
        while fi < nf and fund_min[fi] < m:
            fi += 1
        if fi < nf and fund_min[fi] == m and qty > 0.0 and entry_t < fund_t[fi]:
            f = qty * mo[m] * fund_rate[fi]
            wallet -= f
            fund += f
        # ---- 挂单成交（成交分钟内只检查止损，悲观）----
        filled_now = False
        if pend and m >= pend_start:
            if m >= pend_end:
                pend = False
                o_status[pend_parent] = 2
            else:
                hit = (ml[m] <= pend_lim - pend_tick) if through else (ml[m] <= pend_lim)
                if hit:
                    eqb = wallet
                    efill = pend_lim * (1.0 + slip)
                    notional = eqb * lev
                    if notional > lev_cap * eqb:
                        notional = lev_cap * eqb
                    if notional > 0.0 and efill > sl_h[pend_parent]:
                        k = nt
                        nt += 1
                        qty = notional / efill
                        qty0 = qty
                        fee = notional * maker
                        wallet -= fee
                        entry_px = efill
                        entry_min = m
                        entry_t = open_ns[m] + np.int64(1)     # 成交发生在该子 K 线开盘之后（具体时刻未知）
                        sl = sl_h[pend_parent]
                        tp = tp_h[pend_parent]
                        poc = poc_h[pend_parent]
                        partial_done = False
                        gross = 0.0
                        fees = fee
                        fund = 0.0
                        exit_val = 0.0
                        t_parent[k] = pend_parent
                        t_entry_min[k] = m
                        t_entry_px[k] = efill
                        t_qty[k] = qty
                        t_notional[k] = notional
                        t_eq[k] = eqb
                        t_sl[k] = sl
                        t_tp[k] = tp
                        t_poc[k] = poc
                        o_status[pend_parent] = 1
                        o_fill_min[pend_parent] = m
                        filled_now = True
                    else:
                        o_status[pend_parent] = 2
                    pend = False
        # ---- 持仓离场 ----
        if qty > 0.0 and (filled_now or entry_min < m):
            reason = 0
            if ml[m] <= sl:
                px = (sl if filled_now else min(mo[m], sl)) * (1.0 - slip)
                fee = qty * px * taker
                g = qty * (px - entry_px)
                wallet += g - fee
                gross += g
                fees += fee
                exit_val += qty * px
                qty = 0.0
                reason = 3 if (partial_done and move_be) else 1
            elif not filled_now:
                if tp_mode == 1 and (not partial_done) and poc > entry_px and poc < tp and mh[m] >= poc:
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
                    t_partial_min[k] = m
                    if qty <= 0.0:
                        qty = 0.0
                        reason = 6
                    elif move_be:
                        sl = entry_px
                        if ml[m] <= sl:
                            px = sl * (1.0 - slip)
                            fee = qty * px * taker
                            g = qty * (px - entry_px)
                            wallet += g - fee
                            gross += g
                            fees += fee
                            exit_val += qty * px
                            qty = 0.0
                            reason = 3
                if qty > 0.0 and mh[m] >= tp:
                    px = tp * (1.0 - slip)
                    fee = qty * px * maker
                    g = qty * (px - entry_px)
                    wallet += g - fee
                    gross += g
                    fees += fee
                    exit_val += qty * px
                    qty = 0.0
                    reason = 2
            if qty == 0.0:
                t_exit_min[k] = m
                t_exit_px[k] = exit_val / qty0
                t_fees[k] = fees
                t_fund[k] = fund
                t_gross[k] = gross
                t_pnl[k] = gross - fees - fund
                t_reason[k] = reason
        # ---- 小时收盘：记净值，处理该小时的信号 ----
        last_of_hour = (m == M - 1) or (mhour[m + 1] != h)
        if last_of_hour:
            eq = wallet + qty * (h_close[h] - entry_px) if qty > 0.0 else wallet
            eq_h[h] = eq
            if sig[h]:
                if qty > 0.0 or pend:
                    o_status[h] = 3
                elif limit_mode and lim_h[h] < h_close[h]:
                    pend = True
                    pend_parent = h
                    pend_lim = lim_h[h]
                    pend_tick = tick_h[h]
                    pend_start = m + 1
                    pend_end = m + 1 + valid_min
                else:
                    efill = h_close[h] * (1.0 + slip)
                    notional = eq * lev
                    if notional > lev_cap * eq:
                        notional = lev_cap * eq
                    if notional > 0.0 and eq > 0.0 and efill > sl_h[h]:
                        k = nt
                        nt += 1
                        qty = notional / efill
                        qty0 = qty
                        fee = notional * taker
                        wallet -= fee
                        entry_px = efill
                        entry_min = m            # 收盘成交：从下一根起检查离场
                        entry_t = open_ns[m] + bar_ns          # 入场时刻 = 该子 K 线收盘；恰在该时刻的结算不收（与 1H 引擎一致）
                        sl = sl_h[h]
                        tp = tp_h[h]
                        poc = poc_h[h]
                        partial_done = False
                        gross = 0.0
                        fees = fee
                        fund = 0.0
                        exit_val = 0.0
                        t_parent[k] = h
                        t_entry_min[k] = m
                        t_entry_px[k] = efill
                        t_qty[k] = qty
                        t_notional[k] = notional
                        t_eq[k] = eq
                        t_sl[k] = sl
                        t_tp[k] = tp
                        t_poc[k] = poc
                        o_status[h] = 4 if limit_mode else 1     # 4 = 限价在挂出时即可成交，按市价处理
                        o_fill_min[h] = m
                        eq_h[h] = wallet + qty * (h_close[h] - entry_px)
                    else:
                        o_status[h] = 2
            pos_h[h] = qty
    if qty > 0.0 and k >= 0:
        t_fees[k] = fees
        t_fund[k] = fund
        t_gross[k] = gross + qty * (h_close[n_hours - 1] - entry_px)
        t_pnl[k] = t_gross[k] - fees - fund
        t_reason[k] = 0
    if pend:
        o_status[pend_parent] = 2
    return (eq_h, pos_h, nt, t_parent, t_entry_min, t_exit_min, t_entry_px, t_exit_px, t_qty, t_notional, t_eq, t_sl,
            t_tp, t_poc, t_fees, t_fund, t_gross, t_pnl, t_reason, t_partial_min, o_status, o_fill_min)


def tick_size(index: pd.DatetimeIndex) -> np.ndarray:
    """BTCUSDT 永续的价格最小变动单位，由本地 IS 1H 数据的价格网格推断：最后一根含 0.01 网格价格的 1H K 线开盘于
    2022-02-28 04:00（此后只有 2022-03-21 04:00、2022-04-11 17:00 两根零星例外），因此 2022-02-28 05:00 UTC 之前取 0.01，
    之后取 0.1。"""
    return np.where(index < pd.Timestamp("2022-02-28 05:00", tz="UTC"), 0.01, 0.1)


ORDER_STATUS = {1: "filled", 2: "expired_unfilled", 3: "skipped_position_or_order", 4: "filled_marketable_at_placement"}


def run_path(df1h: pd.DataFrame, signals: pd.DataFrame, bcfg, funding: pd.DataFrame,
             limit_price: np.ndarray | None = None, valid_bars: int = 3, fill_rule: str = "through") -> dict:
    """以 1H K 线为子 K 线运行撮合器（最坏路径规则）。只支持 venue='perp'、sizing='fixed'、复利、无时间止损、无破位离场
    （其余配置直接报错）；在这一配置下市价模式与 1H 引擎逐位一致。资金费按 入场时刻 < 结算时刻 的时间规则收付，
    include_funding=False 时不收；两次结算落在同一根子 K 线时报错（与 backtester.funding_arrays 相同）。"""
    if (bcfg.venue != "perp" or bcfg.sizing != "fixed" or not bcfg.compounding or bcfg.time_stop_bars is not None
            or bcfg.exit_on_breakdown):
        raise ValueError("engine_path supports venue='perp', sizing='fixed', compounding=True, no time stop and no "
                         "breakdown exit only")
    n = len(df1h)
    ht = df1h.index.as_unit("ns").asi8
    if bcfg.funding_on:
        from backtester import check_funding_coverage
        check_funding_coverage(df1h, funding)
        f_t = funding.index.as_unit("ns").asi8
        inwin = (f_t > ht[0]) & (f_t <= ht[-1] + 3600 * 10**9)
        f_t, f_r = f_t[inwin], funding["funding_rate"].to_numpy(dtype=np.float64)[inwin]
    else:
        f_t, f_r = np.zeros(0, dtype=np.int64), np.zeros(0)
    f_bar = np.searchsorted(ht, f_t, side="left")
    keep = f_bar < n
    f_bar, f_r, f_t = f_bar[keep].astype(np.int64), f_r[keep], f_t[keep].astype(np.int64)
    if np.unique(f_bar).size != f_bar.size:
        raise ValueError("two funding settlements map to the same bar (data gap longer than the settlement interval)")
    limit_mode = limit_price is not None
    lim = np.asarray(limit_price, dtype=np.float64) if limit_mode else np.full(n, np.nan)
    out = _sim(df1h["open"].to_numpy(np.float64), df1h["high"].to_numpy(np.float64), df1h["low"].to_numpy(np.float64),
               np.arange(n, dtype=np.int64), df1h["close"].to_numpy(np.float64), n,
               signals["signal"].to_numpy(bool), signals["sl"].to_numpy(np.float64), signals["tp"].to_numpy(np.float64),
               signals["poc_tp"].to_numpy(np.float64), lim, tick_size(df1h.index), int(valid_bars),
               f_bar, f_r, f_t, ht.astype(np.int64), np.int64(3600 * 10**9), float(bcfg.initial_capital), float(bcfg.taker_fee), float(bcfg.maker_fee), float(bcfg.slippage),
               float(bcfg.fixed_leverage), float(bcfg.leverage_cap), {"vah": 0, "poc_partial": 1}[bcfg.tp_mode],
               float(bcfg.partial_frac), bool(bcfg.move_to_breakeven), bool(limit_mode), fill_rule == "through")
    (eq_h, pos_h, nt, par, ebar, xbar, epx, xpx, q, notional, eq, sl, tp, poc, fees, fund, gross, pnl, reason, pbar,
     o_status, o_fill) = out
    nt = int(nt)
    H = pd.Timedelta(hours=1)
    closed = xbar[:nt] >= 0
    market_like = (np.array([o_status[p] == 4 for p in par[:nt]], dtype=bool) if limit_mode
                   else np.ones(nt, dtype=bool))
    trades = pd.DataFrame({
        "parent_bar": par[:nt], "fill_bar": ebar[:nt], "exit_bar": xbar[:nt],
        # 市价单：成交时刻 = 信号 K 线收盘；限价单：成交在 fill_bar 内，具体时刻未知（记录该根收盘作为保守上界）
        "entry_time": [df1h.index[e] + H for e in ebar[:nt]],
        "entry_is_market": market_like,
        "exit_time": [df1h.index[x] + H if x >= 0 else pd.NaT for x in xbar[:nt]],
        "entry_px": epx[:nt], "exit_px": xpx[:nt], "qty": q[:nt], "notional": notional[:nt], "equity_at_entry": eq[:nt],
        "sl": sl[:nt], "tp": tp[:nt], "poc": poc[:nt], "fees": fees[:nt], "funding": fund[:nt], "gross_pnl": gross[:nt],
        "pnl": pnl[:nt], "exit_reason": [REASONS_PATH[int(r)] for r in reason[:nt]], "partial_bar": pbar[:nt],
    })
    trades["entry_bar"] = trades["parent_bar"]
    trades["fill_and_exit_same_bar"] = closed & (trades["fill_bar"] == trades["exit_bar"]) & ~trades["entry_is_market"]
    risk = trades["qty"] * (trades["entry_px"] - trades["sl"])
    trades["r_multiple"] = trades["pnl"] / risk
    trades["ret_on_equity"] = trades["pnl"] / trades["equity_at_entry"]
    trades["hold_bars"] = np.where(closed, xbar[:nt] - ebar[:nt], n - 1 - ebar[:nt])
    idx_close = df1h.index + H
    orders = pd.DataFrame({"status": [ORDER_STATUS.get(int(x), "none") for x in o_status], "fill_bar": o_fill},
                          index=df1h.index)
    return {"equity": pd.Series(eq_h, index=idx_close), "position": pd.Series(pos_h, index=idx_close),
            "trades": trades, "orders": orders[o_status > 0]}


def as_backtest_result(res_path: dict, bcfg, df1h: pd.DataFrame):
    """把路径撮合器的结果包装成 BacktestResult，以复用 metrics 的日度净值与分段指标。"""
    from backtester import BacktestResult
    tr = res_path["trades"].copy()
    tr["ret_on_base"] = tr["ret_on_equity"]
    tr["capped"] = False
    tr["funding_complete"] = True
    return BacktestResult(equity=res_path["equity"], position=res_path["position"], trades=tr, config=bcfg,
                          initial_capital=float(bcfg.initial_capital), start_time=df1h.index[0])

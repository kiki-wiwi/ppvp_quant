"""
ppvp_audit/event_sim.py — 单事件撮合（每个事件独立，不受持仓互斥与复利影响），用于机会级配对比较与匹配随机基准。

规则与 backtester.py / engine_path.py 相同（1H K 线、最坏路径）：
* 入场：市价 = 决策 K 线收盘 × (1 + 滑点)、taker；限价 L（L < 收盘时）自下一根起有效 valid 根，穿价 low ≤ L − tick
  （或触价 low ≤ L）成交于 L × (1 + 滑点)、maker；L ≥ 收盘时按市价处理。成交那一根只检查止损（触及即离场），
  该根的 POC / 止盈不计。
* 出场：止损优先 min(开盘, SL) × (1 − 滑点)、taker；POC 分批（Entry < POC < VAH 时）POC × (1 − 滑点)、maker，
  可选保本上移（同一根回落到保本价按保本离场）；止盈 VAH × (1 − 滑点)、maker；到 end_bar − 1 仍未了结按其收盘离场、taker。
* 输出按 1 单位名义：ret = 净盈亏 / 入场名义（含手续费与滑点，不含资金费），R = 净盈亏 / (入场价 − SL)。
"""
from __future__ import annotations

import numpy as np

from numba_compat import njit


@njit(cache=True)
def _events(o, h, l, c, dec_bar, sl_a, tp_a, poc_a, lim_a, tick_a, valid, through, partial, frac, move_be,
            taker, maker, slip, end_bar):
    m = dec_bar.shape[0]
    filled = np.zeros(m, np.bool_)
    fill_bar = np.full(m, -1, np.int64)
    exit_bar = np.full(m, -1, np.int64)
    reason = np.zeros(m, np.int8)            # 1 SL, 2 TP, 3 BE, 4 未了结（期末收盘）, 5 未成交
    ret = np.zeros(m)
    r_mult = np.full(m, np.nan)
    gross_ref = np.zeros(m)                  # 参考价毛盈亏 / 名义（成本前）
    for j in range(m):
        b = dec_bar[j]
        sl = sl_a[j]
        tp = tp_a[j]
        poc = poc_a[j]
        lim = lim_a[j]
        # ---- 入场 ----
        if lim != lim or lim >= c[b]:
            e_ref = c[b]
            efill = c[b] * (1.0 + slip)
            fee_in = taker
            fb = b
            start = b + 1
        else:
            fb = -1
            last = min(b + valid, end_bar - 1)
            for i in range(b + 1, last + 1):
                hit = (l[i] <= lim - tick_a[j]) if through else (l[i] <= lim)
                if hit:
                    fb = i
                    break
            if fb < 0:
                reason[j] = 5
                continue
            e_ref = lim
            efill = lim * (1.0 + slip)
            fee_in = maker
            start = fb
        if not efill > sl:
            reason[j] = 5
            continue
        filled[j] = True
        fill_bar[j] = fb
        q = 1.0
        stop = sl
        part = False
        pnl = -efill * fee_in
        gref = 0.0
        done = False
        for i in range(start, end_bar):
            first = (i == fb) and (lim == lim and lim < c[b])     # 限价成交的那一根
            if l[i] <= stop:
                px_ref = stop if first else min(o[i], stop)
                px = px_ref * (1.0 - slip)
                pnl += q * (px - efill) - q * px * taker
                gref += q * (px_ref - e_ref)
                exit_bar[j] = i
                reason[j] = 3 if (part and move_be) else 1
                done = True
                break
            if first:
                continue
            if partial and (not part) and poc > efill and poc < tp and h[i] >= poc:
                q1 = q * frac
                px = poc * (1.0 - slip)
                pnl += q1 * (px - efill) - q1 * px * maker
                gref += q1 * (poc - e_ref)
                q -= q1
                part = True
                if move_be:
                    stop = efill
                    if l[i] <= stop:
                        px = stop * (1.0 - slip)
                        pnl += q * (px - efill) - q * px * taker
                        gref += q * (stop - e_ref)
                        exit_bar[j] = i
                        reason[j] = 3
                        done = True
                        break
            if h[i] >= tp:
                px = tp * (1.0 - slip)
                pnl += q * (px - efill) - q * px * maker
                gref += q * (tp - e_ref)
                exit_bar[j] = i
                reason[j] = 2
                done = True
                break
        if not done:
            i = end_bar - 1
            px = c[i] * (1.0 - slip)
            pnl += q * (px - efill) - q * px * taker
            gref += q * (c[i] - e_ref)
            exit_bar[j] = i
            reason[j] = 4
        ret[j] = pnl / efill
        r_mult[j] = pnl / (efill - sl)
        gross_ref[j] = gref / efill
    return filled, fill_bar, exit_bar, reason, ret, r_mult, gross_ref


def simulate_events(df1h, dec_bar, sl, tp, poc=None, limit=None, tick=None, valid: int = 3, fill_rule: str = "through",
                    partial: bool = False, frac: float = 0.5, move_be: bool = False, bcfg=None, end_bar: int | None = None):
    n = len(df1h)
    m = len(dec_bar)
    f = lambda x, fill: np.full(m, fill) if x is None else np.asarray(x, dtype=np.float64)  # noqa: E731
    out = _events(df1h["open"].to_numpy(np.float64), df1h["high"].to_numpy(np.float64), df1h["low"].to_numpy(np.float64),
                  df1h["close"].to_numpy(np.float64), np.asarray(dec_bar, dtype=np.int64), f(sl, np.nan), f(tp, np.nan),
                  f(poc, np.nan), f(limit, np.nan), f(tick, 0.0), int(valid), fill_rule == "through", bool(partial),
                  float(frac), bool(move_be), float(bcfg.taker_fee), float(bcfg.maker_fee), float(bcfg.slippage),
                  int(n if end_bar is None else end_bar))
    keys = ("filled", "fill_bar", "exit_bar", "reason", "ret", "r", "gross_ref_ret")
    return dict(zip(keys, out))

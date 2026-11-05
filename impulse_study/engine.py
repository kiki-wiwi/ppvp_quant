"""
impulse_study/engine.py — Checkpoint 2：多空双向撮合与记账引擎（1m 网格，单品种单仓，numba）。

规则（与 experiment_plan_impulse.md 第 5 节逐条对应；s = +1 多头 / −1 空头）
---------------------------------------------------------------------------
入场   触发 K 线 t（开盘 T，收盘 T + 1min）收盘时冻结 E_ref = C_t、SL 与数量 q_t；下一根 t+1 开盘市价成交，
       Entry_fill = O_{t+1}·(1 + s·slip_entry)，计 taker 费。t+1 为维护占位（bad）时信号作废（features 已排除）。
数量   以损定仓：q = min(R / (max(|E_ref − SL|, 0.0015·E_ref) + c_rt·E_ref), 2.0·Equity_t / E_ref)，R = 0.5%·Equity_t；
       等名义：q = 1.0·Equity_t / E_ref。Equity_t 为 t 收盘时净值（空仓时即现金）。q 不随 O_{t+1} 回改；
       0.0015·E_ref 下限只作用于数量，不移动 SL。
持有   合成 15m 棒以入场时刻为锚：第 1 根 = 1m 棒 t+1 … t+15，第 2 根 = t+16 … t+30。
       第 1 根满（t+15 收盘）：v1 = Σ成交量(t+1..t+15) / 基准中位数，k1 = s·(2C − H − L)/(H − L)；
       v1 ≥ 1.3 且 k1 ≥ 0.10 继续持有，否则按 C_{t+15} 收盘离场；基准 ≤ 0 或 H = L 按不满足处理。
       第 2 根满（t+30 收盘）无条件按 C_{t+30} 离场。B4（fixed1）第 1 根满即离场。
止损   入场棒开盘已越过 SL（多头 O ≤ SL / 空头 O ≥ SL）：保留开仓，按 O_{t+1}·(1 − s·slip_exit) 立即平仓，两边计费；
       此后任一持仓分钟 m：开盘跳空越过 SL → 按 O_m·(1 − s·slip_exit)；否则盘中触及（多头 L ≤ SL / 空头 H ≥ SL）
       → 按 SL·(1 − s·slip_exit)。合成棒末分钟内止损与收盘离场同时满足时，止损优先（保守最坏路径）。
停机   持仓中遇到维护占位分钟：在停机后第一根有效 1m 开盘按 O·(1 − s·slip_exit) 强制离场并计费。
资金费 结算时点 τ（分钟开盘时刻恰为 τ）满足 入场时刻 < τ ≤ 离场时刻 才收付，金额 = −s·q·P_τ·rate（正费率多头付、空头收）。
       P_τ = τ 开始那根 1m 的开盘价；收盘时间离场恰在 τ 时用该分钟收盘价；盘中止损发生在 τ 之前，不收付 τ 的费用。
冷却   离场时刻：开盘类（入场跳空 / 跳空止损 / 停机）= 该分钟开盘；盘中止损与时间离场 = 该分钟收盘。
       下一次入场时刻 ≥ 离场时刻 + 15min，即下一个可接受的触发 K 线索引 ≥ 离场时刻索引 + 14。持仓期间忽略新信号。
盯市   持仓分钟的净值 = 现金 + s·q·(C_m − Entry_fill)（现金已扣入场费并计入已结算资金费）；离场分钟净值 = 平仓后现金。

真实行情锁：Checkpoint 2 只允许合成数据进入撮合（Context.source = 'synthetic'）。REAL_DATA_UNLOCKED 只能在
第 3 阶段经审批的提交中改为 True。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from numba_compat import njit

from . import versions as VS
from .features import Context, VersionInputs

REAL_DATA_UNLOCKED = False

# 离场原因
GAP_ENTRY, GAP_STOP, STOP, TIME1, TIME2, HALT, DATA_END = 1, 2, 3, 4, 5, 6, 7
REASONS = {GAP_ENTRY: "stop_gap_at_entry_open", GAP_STOP: "stop_gap_open", STOP: "stop_intrabar", TIME1: "time_bar1",
           TIME2: "time_bar2", HALT: "halt_forced_exit", DATA_END: "data_end"}
MAX_FUND = 8


class EngineLockedError(RuntimeError):
    """Checkpoint 2：真实行情不得进入撮合。"""


@njit(cache=True)
def _trade_path(i, s, sl, O, H, L, C, V, bad, fund_rate_open, base_med_i, fixed1, v1_min, k1_min, n):
    """从触发 K 线 i 开始的一笔交易（入场棒 e = i + 1）。返回离场与诊断量；fund_unit 为每单位数量的资金费现金流。"""
    e = i + 1
    o_e = O[e]
    fund_b = np.full(MAX_FUND, -1, dtype=np.int64)
    fund_unit = np.zeros(MAX_FUND)
    nf = 0
    mae = 0.0
    mfe = 0.0
    v1 = np.nan
    k1 = np.nan
    cont = -1
    h1 = -np.inf
    l1 = np.inf
    vs = 0.0
    if s * (o_e - sl) <= 0.0:                                   # 入场棒开盘已越过止损
        return e, o_e, GAP_ENTRY, e, e, v1, k1, cont, s * (o_e - o_e), 0.0, nf, fund_b, fund_unit
    j = e
    x_bar = -1
    x_ref = np.nan
    reason = DATA_END
    b_last = -1
    t_exit = -1
    last_px = C[i]
    while True:
        if j >= n:
            x_bar = n - 1
            x_ref = last_px
            reason = DATA_END
            b_last = n - 1
            t_exit = n - 1
            break
        if bad[j]:                                               # 维护停机：停机后第一根有效 1m 开盘强制离场
            v = j
            while v < n and bad[v]:
                v += 1
            if v >= n:
                x_bar = n - 1
                x_ref = last_px
                reason = DATA_END
                b_last = n - 1
                t_exit = n - 1
                break
            x_bar = v
            x_ref = O[v]
            reason = HALT
            b_last = v
            t_exit = v
            break
        if j > e:
            r = fund_rate_open[j]
            if r == r:                                           # 结算时点恰为分钟 j 开盘：入场早于 τ，且尚未离场
                if nf < MAX_FUND:
                    fund_b[nf] = j
                    fund_unit[nf] = -s * O[j] * r
                    nf += 1
            if s * (O[j] - sl) <= 0.0:                           # 开盘跳空越过止损
                x_bar = j
                x_ref = O[j]
                reason = GAP_STOP
                b_last = j
                t_exit = j
                break
        if (s > 0 and L[j] <= sl) or (s < 0 and H[j] >= sl):    # 盘中止损（优先于同一分钟的收盘离场）
            if s * (O[j] - o_e) < mae:                           # 止损分钟只有开盘价确定发生在止损之前；
                mae = s * (O[j] - o_e)                           # 止损之后的价格不计入 MAE / MFE（离场价在下方计入）
            if s * (O[j] - o_e) > mfe:
                mfe = s * (O[j] - o_e)
            x_bar = j
            x_ref = sl
            reason = STOP
            b_last = j
            t_exit = j + 1
            break
        adv = L[j] if s > 0 else H[j]
        fav = H[j] if s > 0 else L[j]
        if s * (adv - o_e) < mae:
            mae = s * (adv - o_e)
        if s * (fav - o_e) > mfe:
            mfe = s * (fav - o_e)
        if H[j] > h1:
            h1 = H[j]
        if L[j] < l1:
            l1 = L[j]
        vs += V[j] if V[j] == V[j] else 0.0
        last_px = C[j]
        k = j - e
        if k == 14:
            if base_med_i > 0.0 and h1 > l1:
                v1 = vs / base_med_i
                k1 = s * (2.0 * C[j] - h1 - l1) / (h1 - l1)
                cont = 1 if (v1 >= v1_min and k1 >= k1_min) else 0
            else:
                cont = 0
            if fixed1 or cont == 0:
                x_bar = j
                x_ref = C[j]
                reason = TIME1
                b_last = j + 1
                t_exit = j + 1
                break
        if k == 29:
            x_bar = j
            x_ref = C[j]
            reason = TIME2
            b_last = j + 1
            t_exit = j + 1
            break
        j += 1
    if (reason == TIME1 or reason == TIME2) and b_last < n:      # 收盘离场恰在结算时点：按该分钟收盘价收付
        r = fund_rate_open[b_last]
        if r == r and nf < MAX_FUND:
            fund_b[nf] = b_last
            fund_unit[nf] = -s * C[x_bar] * r
            nf += 1
    if reason == HALT:                                           # 停机期间（含恢复开盘时刻）的结算
        for b in range(j, x_bar + 1):
            r = fund_rate_open[b]
            if r == r and b > e and nf < MAX_FUND:
                px = O[b] if O[b] == O[b] else last_px
                fund_b[nf] = b
                fund_unit[nf] = -s * px * r
                nf += 1
    if s * (x_ref - o_e) < mae:
        mae = s * (x_ref - o_e)
    if s * (x_ref - o_e) > mfe:
        mfe = s * (x_ref - o_e)
    return x_bar, x_ref, reason, b_last, t_exit, v1, k1, cont, mae, mfe, nf, fund_b, fund_unit


@njit(cache=True)
def _run_portfolio(entry_dir, sl_arr, eref_arr, O, H, L, C, V, bad, fund_rate_open, base_med, start_i, last_i, end_i,
                   fixed1, v1_min, k1_min, fee, slip_e, slip_x, use_funding, risk_mode, risk_frac, min_dist_frac, lev_cap,
                   notional_mult, c_rt, init_cap, cooldown, rec):
    n = C.shape[0]
    equity = np.full(n, np.nan)
    cash = init_cap
    i = start_i
    m_fill = start_i
    next_ok = start_i
    ntr = 0
    while i <= last_i:
        d = entry_dir[i]
        if d == 0 or i < next_ok:
            i += 1
            continue
        s = float(d)
        sl = sl_arr[i]
        eref = eref_arr[i]
        for m in range(m_fill, i + 1):
            equity[m] = cash
        eq_t = cash
        dist = abs(eref - sl)
        if risk_mode:
            q = min(risk_frac * eq_t / (max(dist, min_dist_frac * eref) + c_rt * eref), lev_cap * eq_t / eref)
        else:
            q = notional_mult * eq_t / eref
        x_bar, x_ref, reason, b_last, t_exit, v1, k1, cont, mae, mfe, nf, fund_b, fund_unit = _trade_path(
            i, s, sl, O, H, L, C, V, bad, fund_rate_open, base_med[i], fixed1, v1_min, k1_min, n)
        e = i + 1
        entry_fill = O[e] * (1.0 + s * slip_e)
        fee_e = fee * q * entry_fill
        cash -= fee_e
        funding = 0.0
        fpos = 0
        last_px = C[i]
        for m in range(e, x_bar):
            while fpos < nf and fund_b[fpos] <= m:
                if use_funding:
                    cash += q * fund_unit[fpos]
                    funding += q * fund_unit[fpos]
                fpos += 1
            px = C[m] if C[m] == C[m] else last_px
            last_px = px
            equity[m] = cash + s * q * (px - entry_fill)
        while fpos < nf:
            if use_funding:
                cash += q * fund_unit[fpos]
                funding += q * fund_unit[fpos]
            fpos += 1
        exit_fill = x_ref * (1.0 - s * slip_x)
        fee_x = fee * q * exit_fill
        cash += s * q * (exit_fill - entry_fill) - fee_x
        equity[x_bar] = cash
        rec[ntr, 0] = i
        rec[ntr, 1] = e
        rec[ntr, 2] = x_bar
        rec[ntr, 3] = s
        rec[ntr, 4] = reason
        rec[ntr, 5] = sl
        rec[ntr, 6] = eref
        rec[ntr, 7] = q
        rec[ntr, 8] = O[e]
        rec[ntr, 9] = entry_fill
        rec[ntr, 10] = x_ref
        rec[ntr, 11] = exit_fill
        rec[ntr, 12] = fee_e
        rec[ntr, 13] = fee_x
        rec[ntr, 14] = funding
        rec[ntr, 15] = s * q * (x_ref - O[e])                         # 参考价毛盈亏（无滑点、手续费、资金费）
        rec[ntr, 16] = s * q * (exit_fill - entry_fill) - fee_e - fee_x + funding
        rec[ntr, 17] = v1
        rec[ntr, 18] = k1
        rec[ntr, 19] = cont
        rec[ntr, 20] = mae
        rec[ntr, 21] = mfe
        rec[ntr, 22] = nf
        rec[ntr, 23] = eq_t
        rec[ntr, 24] = t_exit
        rec[ntr, 25] = b_last
        ntr += 1
        next_ok = t_exit + cooldown - 1
        m_fill = x_bar + 1
        nxt = x_bar + 1 if x_bar + 1 > i + 1 else i + 1
        i = nxt if nxt > next_ok else next_ok
    for m in range(m_fill, end_i + 1):
        equity[m] = cash
    return ntr, equity


@njit(cache=True)
def _simulate_events(idx, s_arr, sl_arr, eref_arr, O, H, L, C, V, bad, fund_rate_open, base_med, fixed1, v1_min, k1_min,
                     fee, slip_e, slip_x, use_funding, out):
    """互不影响的独立事件（匹配随机对照用）：每个事件按同一撮合规则独立模拟，数量取 1 单位，结果按 E_ref 归一为 bp。"""
    n = C.shape[0]
    for k in range(idx.shape[0]):
        i = idx[k]
        s = s_arr[k]
        sl = sl_arr[k]
        eref = eref_arr[k]
        x_bar, x_ref, reason, b_last, t_exit, v1, k1, cont, mae, mfe, nf, fund_b, fund_unit = _trade_path(
            i, s, sl, O, H, L, C, V, bad, fund_rate_open, base_med[i], fixed1, v1_min, k1_min, n)
        e = i + 1
        entry_fill = O[e] * (1.0 + s * slip_e)
        exit_fill = x_ref * (1.0 - s * slip_x)
        funding = 0.0
        if use_funding:
            for f in range(nf):
                funding += fund_unit[f]
        net = s * (exit_fill - entry_fill) - fee * (entry_fill + exit_fill) + funding
        out[k, 0] = s * (x_ref - O[e]) / eref * 1e4
        out[k, 1] = net / eref * 1e4
        out[k, 2] = net / abs(eref - sl)
        out[k, 3] = reason
        out[k, 4] = x_bar
        out[k, 5] = cont
        out[k, 6] = mae / eref * 1e4
        out[k, 7] = mfe / eref * 1e4


REC_COLS = ["trigger_idx", "entry_idx", "exit_bar", "side", "reason", "sl", "e_ref", "qty", "entry_open", "entry_fill",
            "exit_ref", "exit_fill", "fee_entry", "fee_exit", "funding", "gross_pnl", "net_pnl", "v1", "k1", "cont",
            "mae_px", "mfe_px", "n_funding", "equity_at_signal", "exit_time_idx", "funding_last_boundary"]


@dataclass
class PortfolioResult:
    trades: pd.DataFrame
    equity: pd.Series            # 逐分钟盯市净值（窗口之外为 NaN）
    daily_equity: pd.Series      # UTC 日末净值
    meta: dict


def _guard(ctx: Context) -> None:
    if ctx.source != "synthetic" and not REAL_DATA_UNLOCKED:
        raise EngineLockedError(f"Checkpoint 2: data source {ctx.source!r} may not enter the matching engine")


def _window_bounds(ctx: Context) -> tuple[int, int, int]:
    w = np.flatnonzero(ctx.window)
    if not len(w):
        raise ValueError("empty trading window")
    start_i, last_i = int(w[0]), int(w[-1])
    end_i = len(ctx.C) - 1
    if last_i + 30 > end_i:                      # 最长持仓的最后一分钟 t+30 必须在网格内（IS：23:29 触发 → 23:59）
        raise ValueError("data must extend at least 30 minutes beyond the last eligible trigger")
    return start_i, last_i, end_i


def run_portfolio(ctx: Context, vin: VersionInputs, cost: VS.CostScenario, sizing: VS.Sizing,
                  initial_capital: float = VS.INITIAL_CAPITAL) -> PortfolioResult:
    _guard(ctx)
    start_i, last_i, end_i = _window_bounds(ctx)
    rec = np.full((int((vin.entry_dir != 0).sum()) + 1, len(REC_COLS)), np.nan)
    ntr, equity = _run_portfolio(
        vin.entry_dir, vin.sl, vin.eref, ctx.O, ctx.H, ctx.L, ctx.C, ctx.V, ctx.bad, ctx.funding_rate_open, ctx.base_med,
        start_i, last_i, end_i, vin.spec.hold == "fixed1", VS.V1_MIN, VS.K1_MIN, cost.fee_taker, cost.slip_entry,
        cost.slip_exit, cost.include_funding, sizing.mode == "risk", sizing.risk_fraction, sizing.min_dist_frac,
        sizing.leverage_cap, sizing.notional_mult, cost.c_rt, float(initial_capital), VS.COOLDOWN_MIN, rec)
    tr = pd.DataFrame(rec[:ntr], columns=REC_COLS)
    for c in ("trigger_idx", "entry_idx", "exit_bar", "reason", "n_funding", "exit_time_idx", "funding_last_boundary"):
        tr[c] = tr[c].astype(np.int64)
    tr["side"] = tr["side"].astype(np.int64)
    tr = _enrich(ctx, vin, tr)
    eq = pd.Series(equity, index=ctx.index, name="equity")
    day_end = eq.dropna()
    daily = day_end.groupby(day_end.index.floor("D")).last()
    return PortfolioResult(tr, eq, daily, {"version": vin.spec.name, "cost": cost.name, "sizing": sizing.name,
                                           "start_idx": start_i, "last_trigger_idx": last_i, "n_trades": ntr,
                                           "final_equity": float(equity[end_i]), "skipped_signals": vin.skipped})


def _enrich(ctx: Context, vin: VersionInputs, tr: pd.DataFrame) -> pd.DataFrame:
    if not len(tr):
        return tr
    idx = ctx.index
    i, e, x = tr["trigger_idx"].to_numpy(), tr["entry_idx"].to_numpy(), tr["exit_bar"].to_numpy()
    t0 = idx[0]
    tr.insert(0, "trigger_time", idx[i])
    tr.insert(1, "entry_time", idx[e])
    tr["exit_time"] = t0 + pd.to_timedelta(tr["exit_time_idx"].to_numpy(), unit="min")     # 完整分钟网格：索引即分钟数
    tr["reason_name"] = tr["reason"].map(REASONS)
    tr["category"] = vin.cat[i]
    tr["w"] = vin.w[i]
    tr["atr15"] = vin.atr_at[i]
    tr["stop_dist_frac"] = (tr["e_ref"] - tr["sl"]).abs() / tr["e_ref"]
    tr["risk_amount"] = tr["qty"] * (tr["e_ref"] - tr["sl"]).abs()
    tr["leverage_at_eref"] = tr["qty"] * tr["e_ref"] / tr["equity_at_signal"]
    tr["gross_bps"] = tr["gross_pnl"] / (tr["qty"] * tr["e_ref"]) * 1e4
    tr["net_bps"] = tr["net_pnl"] / (tr["qty"] * tr["e_ref"]) * 1e4
    tr["net_R"] = tr["net_pnl"] / tr["risk_amount"]
    tr["bars_held"] = x - e + 1
    gcs = np.concatenate([[0], np.cumsum(ctx.archive_gap.astype(np.int64))])
    tr["has_archive_gap"] = (gcs[x + 1] - gcs[e]) > 0
    lo = np.maximum(i - 15, 0)
    tr["signal_lookback_has_archive_gap"] = (gcs[i + 1] - gcs[lo]) > 0
    bcs = np.concatenate([[0], np.cumsum(ctx.bad.astype(np.int64))])
    tr["halt_in_holding"] = (bcs[x] - bcs[e]) > 0
    return tr


def simulate_events(ctx: Context, idx: np.ndarray, side: np.ndarray, sl: np.ndarray, eref: np.ndarray, fixed1: bool,
                    cost: VS.CostScenario) -> pd.DataFrame:
    """独立事件模拟（第 3 阶段匹配随机对照用；不经过组合的持仓互斥与冷却）。"""
    _guard(ctx)
    out = np.full((len(idx), 8), np.nan)
    _simulate_events(np.asarray(idx, np.int64), np.asarray(side, float), np.asarray(sl, float), np.asarray(eref, float),
                     ctx.O, ctx.H, ctx.L, ctx.C, ctx.V, ctx.bad, ctx.funding_rate_open, ctx.base_med, fixed1, VS.V1_MIN,
                     VS.K1_MIN, cost.fee_taker, cost.slip_entry, cost.slip_exit, cost.include_funding, out)
    return pd.DataFrame(out, columns=["gross_bps", "net_bps", "net_R", "reason", "exit_bar", "cont", "mae_bps", "mfe_bps"])

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
停机   持仓中遇到维护占位分钟：在停机后第一根有效 1m 开盘按 O·(1 − s·slip_exit) 强制离场并计费（该开盘即使已越过 SL
       也记为 halt_forced_exit；停机可使持仓超过 30 分钟）。停机延续到数据末端时没有定义的离场，引擎直接报错。
资金费 结算时点 τ（分钟开盘时刻恰为 τ）满足 入场时刻 < τ ≤ 离场时刻 才收付，金额 = −s·q·P_τ·rate（正费率多头付、空头收），
       结算次数不设上限。P_τ = τ 开始那根 1m 的开盘价（停机占位分钟取其开盘价，即停机前最后收盘价）；收盘时间离场恰在
       τ 时用该分钟收盘价；盘中止损发生在 τ 之前，不收付 τ 的费用。
冷却   离场时刻：开盘类（入场跳空 / 跳空止损 / 停机）= 该分钟开盘；盘中止损与时间离场 = 该分钟收盘。
       下一次入场时刻 ≥ 离场时刻 + 15min，即下一个可接受的触发 K 线索引 ≥ 离场时刻索引 + 14。持仓期间忽略新信号。
盯市   持仓分钟的净值 = 现金 + s·q·(C_m − Entry_fill)（现金已扣入场费，并已计入 τ ≤ 该分钟开盘的资金费）；离场分钟净值 =
       平仓后现金（含恰在离场收盘时刻 τ 的结算）。
台账   cont 为第 1 根合成棒满时延续检验的结果：1 满足、0 不满足（含 B_t ≤ 0 或 H = L）、−1 未检验（在 t+15 收盘之前已离场，
       或停机覆盖 t+15）；B4 照常检验、只作诊断。v1、k1 各自在可计算时记录（B_t > 0 / H > L）。以损定仓时 cap_bound =
       2.0x 上限严格小于以损定仓数量（上限决定了数量），floor_bound = |E_ref − SL| 严格小于 0.15%·E_ref（下限在仓位分母中
       生效，不论上限是否同时约束）；恰好相等记为假，等名义口径恒为假。funding_last_boundary 为资金费窗口内最后一个分钟开盘
       的索引（诊断列）。
输入   版本输入必须由同一个上下文对象构造（features.build_version_inputs / synthetic.manual_inputs 记录 vin.context），
       构造之后该上下文的交易窗口、该版本的触发条件与（A 组）箱体没有被原地改动（vin.input_key），
       各数组为一维且与分钟网格等长；entry_dir 整列只能取 −1 / 0 / +1；窗口内信号的入场 K 线有效、E_ref 与 SL 有限且 SL
       位于亏损一侧、基准成交量有定义；交易窗口最后一个可触发分钟之后至少还有 30 分钟数据。不满足即报错，不静默处理。

真实行情锁：Checkpoint 2 只允许合成数据进入撮合（Context.source = 'synthetic'）。合成标签在 features.make_context 构造上下文时
核对（网格必须带 synthetic.grid_from_arrays 写入的合成数据标记）；dataclasses.replace 仍可显式改写标签——锁防的是误用，
不防有意改标签。REAL_DATA_UNLOCKED 只能在第 3 阶段经审批的提交中改为 True；解锁后真实上下文还必须带有完整的资金费结算
（网格内全部 UTC 00:00 / 08:00 / 16:00 分钟，IS 全网格 4,383 次，不多不少）与冻结的漏记分钟清单。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from numba_compat import njit

from . import versions as VS
from .features import Context, VersionInputs, input_key

REAL_DATA_UNLOCKED = True

# 离场原因（计划没有定义“数据末端”离场：持仓走到网格末端仍未离场时引擎直接报错）
GAP_ENTRY, GAP_STOP, STOP, TIME1, TIME2, HALT = 1, 2, 3, 4, 5, 6
REASONS = {GAP_ENTRY: "stop_gap_at_entry_open", GAP_STOP: "stop_gap_open", STOP: "stop_intrabar", TIME1: "time_bar1",
           TIME2: "time_bar2", HALT: "halt_forced_exit"}
TIME_EXIT_REASONS = (TIME1, TIME2)


class EngineLockedError(RuntimeError):
    """Checkpoint 2：真实行情不得进入撮合。"""


@njit(cache=True)
def _trade_path(i, s, sl, O, H, L, C, V, bad, fund_rate_open, base_med_i, fixed1, v1_min, k1_min, n):
    """从触发 K 线 i 开始的一笔交易（入场棒 e = i + 1）。

    先按价格路径确定离场，再一次性收集资金费窗口 入场时刻 < τ ≤ 离场时刻 内的全部结算（不设次数上限）。
    返回离场与诊断量；fund_b 为结算所在分钟（升序），fund_unit 为每单位数量的资金费现金流。
    """
    e = i + 1
    o_e = O[e]
    mae = 0.0
    mfe = 0.0
    v1 = np.nan
    k1 = np.nan
    cont = -1
    h1 = -np.inf
    l1 = np.inf
    vs = 0.0
    if s * (o_e - sl) <= 0.0:                                   # 入场棒开盘已越过止损：开仓即平仓，τ = 入场时刻不收付
        return (e, o_e, GAP_ENTRY, e, e, v1, k1, cont, 0.0, 0.0, 0, np.zeros(0, dtype=np.int64), np.zeros(0))
    j = e
    x_bar = -1
    x_ref = np.nan
    reason = 0
    b_last = -1
    t_exit = -1
    last_px = C[i]                                               # 最近一根已处理有效分钟的收盘价
    while True:
        if j >= n:
            raise ValueError("a trade reached the end of the minute grid before any exit rule fired")
        if bad[j]:                                               # 维护停机：停机后第一根有效 1m 开盘强制离场
            v = j
            while v < n and bad[v]:
                v += 1
            if v >= n:
                raise ValueError("a maintenance halt during a holding runs to the end of the minute grid")
            x_bar = v
            x_ref = O[v]
            reason = HALT
            b_last = v
            t_exit = v
            break
        if j > e and s * (O[j] - sl) <= 0.0:                     # 开盘跳空越过止损
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
        if k == 14:                                              # 第 1 根合成棒满：延续检验（B4 也检验，只作诊断）
            ok_v = base_med_i > 0.0
            ok_k = h1 > l1
            if ok_v:
                v1 = vs / base_med_i
            if ok_k:
                k1 = s * (2.0 * C[j] - h1 - l1) / (h1 - l1)
            cont = 1 if (ok_v and ok_k and v1 >= v1_min and k1 >= k1_min) else 0
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
    # 资金费：持仓中经过的每个分钟开盘 τ ∈ (入场时刻, 离场分钟开盘]（含跳空止损、盘中止损与停机恢复的那一分钟），
    # 时间离场另加恰在离场收盘时刻的 τ = 离场分钟 + 1（按该分钟收盘价）。
    t_close = (reason == TIME1 or reason == TIME2) and b_last < n and fund_rate_open[b_last] == fund_rate_open[b_last]
    nf = 1 if t_close else 0
    for b in range(e + 1, x_bar + 1):
        if fund_rate_open[b] == fund_rate_open[b]:
            nf += 1
    fund_b = np.empty(nf, dtype=np.int64)
    fund_unit = np.empty(nf)
    f = 0
    for b in range(e + 1, x_bar + 1):
        r = fund_rate_open[b]
        if r == r:
            px = O[b] if O[b] == O[b] else last_px               # 停机占位分钟的开盘价即停机前最后收盘价
            fund_b[f] = b
            fund_unit[f] = -s * px * r
            f += 1
    if t_close:                                                  # 收盘离场恰在结算时点：按该分钟收盘价收付
        fund_b[f] = b_last
        fund_unit[f] = -s * C[x_bar] * fund_rate_open[b_last]
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
        cap_bound = 0
        floor_bound = 0
        if risk_mode:
            floor_d = min_dist_frac * eref
            if dist < floor_d:
                floor_bound = 1                                      # 下限只限制数量，SL 不动
            q_risk = risk_frac * eq_t / (max(dist, floor_d) + c_rt * eref)
            q_cap = lev_cap * eq_t / eref
            if q_cap < q_risk:
                cap_bound = 1
            q = min(q_risk, q_cap)
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
        rec[ntr, 26] = cap_bound
        rec[ntr, 27] = floor_bound
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
            "mae_px", "mfe_px", "n_funding", "equity_at_signal", "exit_time_idx", "funding_last_boundary", "cap_bound",
            "floor_bound"]
_INT_COLS = ("trigger_idx", "entry_idx", "exit_bar", "side", "reason", "cont", "n_funding", "exit_time_idx",
             "funding_last_boundary")
EVENT_COLS = ["gross_bps", "net_bps", "net_R", "reason", "exit_bar", "cont", "mae_bps", "mfe_bps"]


@dataclass
class PortfolioResult:
    trades: pd.DataFrame
    equity: pd.Series            # 逐分钟盯市净值（窗口之外为 NaN）
    daily_equity: pd.Series      # UTC 日末净值
    meta: dict


_SETTLEMENT_NS = 8 * 3_600_000_000_000          # Binance USDⓈ-M BTCUSDT 在 IS 内的结算间隔：UTC 00:00 / 08:00 / 16:00


def _require_real_inputs(ctx: Context) -> None:
    """解锁之后真实上下文的输入完整性（实验计划第 10 节）：
    1. 资金费：构造时传入了资金费序列，且映射到网格上的结算恰为网格内全部 UTC 00:00 / 08:00 / 16:00 分钟（不多不少；
       IS 全网格即 4,383 次。原始 calc_time 的毫秒抖动须先由运行脚本取整到分钟）；
    2. 漏记分钟：网格范围内标记的漏记分钟恰为 versions.ARCHIVE_GAP_MINUTES 中落在网格内的分钟。"""
    ns = np.asarray(ctx.index.asi8)
    sched = ns[ns % _SETTLEMENT_NS == 0]
    if not ctx.funding_supplied or not np.array_equal(ns[np.isfinite(ctx.funding_rate_open)], sched):
        raise ValueError("a real-data context must carry a funding settlement at exactly the UTC 00:00/08:00/16:00 "
                         "minutes of the grid")
    want = pd.DatetimeIndex(pd.to_datetime(list(VS.ARCHIVE_GAP_MINUTES), utc=True)).as_unit("ns").asi8
    want = np.sort(want[(want >= ns[0]) & (want <= ns[-1])])
    if not np.array_equal(np.sort(ns[ctx.archive_gap]), want):
        raise ValueError("a real-data context must flag exactly the frozen archive-gap minutes (versions.ARCHIVE_GAP_MINUTES)")


def _guard(ctx: Context) -> None:
    if ctx.source == "synthetic":
        return                                   # make_context 构造时已核对网格带合成数据标记（见模块说明）
    if not REAL_DATA_UNLOCKED:
        raise EngineLockedError(f"Checkpoint 2: data source {ctx.source!r} may not enter the matching engine")
    _require_real_inputs(ctx)


def _window_bounds(ctx: Context) -> tuple[int, int, int]:
    w = np.flatnonzero(ctx.window)
    if not len(w):
        raise ValueError("empty trading window")
    start_i, last_i = int(w[0]), int(w[-1])
    end_i = len(ctx.C) - 1
    if last_i + 30 > end_i:                      # 最长持仓的最后一分钟 t+30 必须在网格内（IS：23:29 触发 → 23:59）
        raise ValueError("data must extend at least 30 minutes beyond the last eligible trigger")
    return start_i, last_i, end_i


def _check_signals(ctx: Context, i: np.ndarray, side: np.ndarray, sl: np.ndarray, eref: np.ndarray) -> None:
    """引擎层面的输入合法性（版本输入与匹配随机候选都必须满足）：方向 ±1、入场 K 线 t+1 在网格内且有效、
    E_ref 与 SL 有限且 SL 位于亏损一侧、基准成交量有定义。"""
    n = len(ctx.C)
    if len(i) and (i.min() < 0 or i.max() + 1 > n - 1):
        raise ValueError("signal minute outside the grid (the entry bar t+1 must exist)")
    if not np.isin(side, (-1.0, 1.0)).all():
        raise ValueError("side must be +1 or -1")
    if ctx.bad[i + 1].any():
        raise ValueError("signal whose entry bar t+1 is a maintenance placeholder (bad); such signals are void")
    if not (np.isfinite(eref).all() and np.isfinite(sl).all() and (side * (eref - sl) > 0).all()):
        raise ValueError("stop must be finite and on the losing side of E_ref")
    if not np.isfinite(ctx.base_med[i]).all():
        raise ValueError("baseline volume undefined at a signal minute")


_VIN_ARRAYS = ("entry_dir", "sl", "eref", "w", "cat", "atr_at")


def _check_version_inputs(ctx: Context, vin: VersionInputs) -> None:
    """版本输入必须由同一个上下文对象构造（换窗口、换触发条件或箱体的上下文都必须重建版本输入），构造之后该上下文的
    交易窗口、该版本阈值口径的触发条件与（A 组）箱体没有被原地改动（features.input_key），且每个数组都是一维、
    与分钟网格等长（撮合核按分钟索引直接读取，长度不符会越界读写）。"""
    if vin.context is not ctx:
        raise ValueError("version inputs were built on a different context; rebuild them with "
                         "features.build_version_inputs on this context")
    if vin.input_key is None or vin.input_key != input_key(ctx, vin.spec):
        raise ValueError("the context's trading window, trigger conditions or box changed in place after the version "
                         "inputs were built; rebuild them with features.build_version_inputs on this context")
    n = len(ctx.C)
    for f in _VIN_ARRAYS:
        a = getattr(vin, f)
        if not isinstance(a, np.ndarray) or a.ndim != 1 or a.shape[0] != n:
            raise ValueError(f"VersionInputs.{f} must be a 1-D array with one entry per grid minute ({n})")


def run_portfolio(ctx: Context, vin: VersionInputs, cost: VS.CostScenario, sizing: VS.Sizing,
                  initial_capital: float = VS.INITIAL_CAPITAL) -> PortfolioResult:
    _guard(ctx)
    start_i, last_i, end_i = _window_bounds(ctx)
    _check_version_inputs(ctx, vin)
    if not np.isin(vin.entry_dir, (-1, 0, 1)).all():
        raise ValueError("entry_dir must be -1, 0 or +1")
    sig = np.flatnonzero(vin.entry_dir != 0)
    sig = sig[(sig >= start_i) & (sig <= last_i)]                 # 窗口之外的信号不进入撮合
    _check_signals(ctx, sig, vin.entry_dir[sig].astype(float), vin.sl[sig], vin.eref[sig])
    rec = np.full((len(sig) + 1, len(REC_COLS)), np.nan)
    ntr, equity = _run_portfolio(
        vin.entry_dir, vin.sl, vin.eref, ctx.O, ctx.H, ctx.L, ctx.C, ctx.V, ctx.bad, ctx.funding_rate_open, ctx.base_med,
        start_i, last_i, end_i, vin.spec.hold == "fixed1", VS.V1_MIN, VS.K1_MIN, cost.fee_taker, cost.slip_entry,
        cost.slip_exit, cost.include_funding, sizing.mode == "risk", sizing.risk_fraction, sizing.min_dist_frac,
        sizing.leverage_cap, sizing.notional_mult, cost.c_rt, float(initial_capital), VS.COOLDOWN_MIN, rec)
    tr = pd.DataFrame(rec[:ntr], columns=REC_COLS)
    for c in _INT_COLS:
        tr[c] = tr[c].astype(np.int64)
    for c in ("cap_bound", "floor_bound"):
        tr[c] = tr[c].astype(bool)
    tr = _enrich(ctx, vin, tr)
    eq = pd.Series(equity, index=ctx.index, name="equity")
    day_end = eq.dropna()
    daily = day_end.groupby(day_end.index.floor("D")).last()
    return PortfolioResult(tr, eq, daily, {"version": vin.spec.name, "cost": cost.name, "sizing": sizing.name,
                                           "start_idx": start_i, "last_trigger_idx": last_i, "n_trades": ntr,
                                           "n_signals_in_window": int(len(sig)), "final_equity": float(equity[end_i]),
                                           "eligibility_exclusions": vin.skipped,
                                           "eligibility_exclusions_exclusive": vin.skipped_exclusive})


def _enrich(ctx: Context, vin: VersionInputs, tr: pd.DataFrame) -> pd.DataFrame:
    """派生列；空台账也得到完整的列与类型（零笔成交的版本照常汇总为 n = 0）。"""
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
    tr["bars_held"] = x - e + 1                    # t+1 至离场分钟（含）的 1m 根数，与 has_archive_gap 的覆盖范围相同
    tr["minutes_held"] = tr["exit_time_idx"].to_numpy() - e     # 持仓时长（分钟）：入场开盘至离场时刻；入场开盘跳空离场为 0
    gcs = np.concatenate([[0], np.cumsum(ctx.archive_gap.astype(np.int64))])
    tr["has_archive_gap"] = (gcs[x + 1] - gcs[e]) > 0
    lo = np.maximum(i - 15, 0)
    tr["signal_lookback_has_archive_gap"] = (gcs[i + 1] - gcs[lo]) > 0
    bcs = np.concatenate([[0], np.cumsum(ctx.bad.astype(np.int64))])
    tr["halt_in_holding"] = (bcs[x] - bcs[e]) > 0
    return tr


def simulate_events(ctx: Context, idx: np.ndarray, side: np.ndarray, sl: np.ndarray, eref: np.ndarray, fixed1: bool,
                    cost: VS.CostScenario) -> pd.DataFrame:
    """独立事件模拟（第 3 阶段匹配随机对照与图 4 起点对照用；不经过组合的持仓互斥与冷却）。

    每个事件都必须是合法的入场：触发分钟在上下文的交易窗口内、没有维护污染、入场 K 线有效、基准成交量有定义、
    止损位于亏损一侧，且网格在触发后至少还有 30 分钟（与 run_portfolio 的窗口保护相同）；否则直接报错。
    """
    _guard(ctx)
    idx = np.asarray(idx)
    if idx.size and not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("event indices must be integers")
    idx = idx.astype(np.int64)
    side, sl, eref = (np.asarray(a, dtype=float) for a in (side, sl, eref))
    if not (len(idx) == len(side) == len(sl) == len(eref)):
        raise ValueError("idx, side, sl and eref must have the same length")
    n = len(ctx.C)
    if len(idx) and (idx.min() < 0 or idx.max() + 30 > n - 1):
        raise ValueError("event minute must leave at least 30 minutes of data after it")
    if len(idx) and not ctx.window[idx].all():
        raise ValueError("event minute outside the context's trading window")
    if len(idx) and ctx.frame["contaminated"].to_numpy()[idx].any():
        raise ValueError("event minute is maintenance-contaminated")
    _check_signals(ctx, idx, side, sl, eref)
    out = np.full((len(idx), len(EVENT_COLS)), np.nan)
    _simulate_events(idx, side, sl, eref, ctx.O, ctx.H, ctx.L, ctx.C, ctx.V, ctx.bad, ctx.funding_rate_open,
                     ctx.base_med, fixed1, VS.V1_MIN, VS.K1_MIN, cost.fee_taker, cost.slip_entry, cost.slip_exit,
                     cost.include_funding, out)
    return pd.DataFrame(out, columns=EVENT_COLS)

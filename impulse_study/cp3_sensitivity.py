"""Pre-unlock implementations of frozen plan section 11(1--9).

This module never loads market data, mutates a Context, or changes the frozen
engine. Its default CP3 entry point runs only G8z and the B full-period appendix.
CP4 is an explicit separate entry point, whose statistics are supplied as hooks.

Choices not further specified by the plan, fixed before any real result:
* Empty expanding histories produce NaN (no Z trigger); valid sigma includes
  finite zero values and all minutes, including contaminated/bad minutes.
  Two heaps maintain the exact nanmedian in O(n log n), sampled BEFORE each UTC
  day's values are inserted. The lower bound is 2020-01-01 00:15, inclusive.
  New temporal helpers normalize aware indices to UTC nanoseconds before integer
  comparisons, as approved after the microsecond synthetic-fixture failure.
* Defect removal removes the UNION of the three flags. Surviving trades retain
  their original quantity and schedule; skipped opportunities are not reopened.
  Their fees/funding/marked equity are rebuilt from the retained ledger.
* Optimistic and natural-bar reruns use main cost/risk sizing for section 11's
  descriptive comparisons. Defect removal also supplies the harsh-cost ledger
  required by G6; section 6 exposes all 80 frozen configurations unchanged.
* Stop/close conflicts count simultaneous eligible exits, including an equal
  reference price; a separate economic_result_changes flag identifies a changed
  per-unit net result (including a newly payable close-boundary funding charge).
* Natural-bar baseline sums bad-minute volume as zero, just like the main
  baseline. Complete natural bars include minutes before entry, while MAE/MFE
  and P&L start at entry. Optimistic closes use the whole exit-minute H/L for
  MAE/MFE because the position now survives through that minute's close.
* Full-period appendix and box variants rebuild Context with make_context;
  only causal Z uses replace(new cond), as explicitly required in section 5.8.
  Fresh reconstruction preserves a supplied empty funding series distinctly.
* The alternative matching/accounting kernel below is copied from the frozen
  engine; only holding boundaries, continuation inputs and stop/close priority
  differ. Ledger equity reconstruction is a separate cash-event calculation.
"""
from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import heapq
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from numba_compat import njit
from . import engine as E
from . import features as F
from . import signals as S
from . import versions as VS
from .scope import IS_START, MINUTE_NS

Z_VERSIONS = ("A5_box_origin_z", "B5_pure_bi_mom_z")
B_VERSIONS = tuple(k for k, v in VS.VERSIONS.items() if v.group == "B")
BOX_VERSIONS = tuple(k for k, v in VS.VERSIONS.items() if v.group == "A" and v.trigger_rule == "pct")
DEFECT_FLAGS = ("has_archive_gap", "signal_lookback_has_archive_gap", "halt_in_holding")
FIRST_Z = pd.Timestamp("2020-01-07 15:51", tz="UTC")
GAP_ENTRY, GAP_STOP, STOP, TIME1, TIME2, HALT = E.GAP_ENTRY, E.GAP_STOP, E.STOP, E.TIME1, E.TIME2, E.HALT


def context_fingerprint(ctx: F.Context) -> str:
    """Bitwise content digest of every Context field, useful for immutability tests."""
    h = hashlib.sha256()

    def visit(value):
        h.update(type(value).__name__.encode())
        if isinstance(value, np.ndarray):
            h.update(str(value.dtype).encode())
            h.update(str(value.shape).encode())
            h.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, pd.DataFrame):
            visit(value.index)
            for name in value.columns:
                h.update(str(name).encode())
                visit(value[name].to_numpy())
        elif isinstance(value, pd.DatetimeIndex):
            h.update(str(value.tz).encode())
            visit(value.asi8)
        elif isinstance(value, dict):
            for key in sorted(value):
                h.update(str(key).encode())
                visit(value[key])
        else:
            h.update(repr(value).encode())
    for field in fields(ctx):
        h.update(field.name.encode())
        visit(getattr(ctx, field.name))
    return h.hexdigest()


def causal_sigma_reference(ctx: F.Context) -> pd.Series:
    """Day-D median over finite sigma(T), 2020-01-01 00:15 <= T < D 00:00."""
    index = pd.DatetimeIndex(ctx.index).tz_convert("UTC").as_unit("ns")
    ns = index.asi8
    days = index.floor("D").asi8
    sigma = ctx.frame["sigma7d"].to_numpy(dtype=float)
    lower = int((IS_START + pd.Timedelta(minutes=15)).value)
    out = np.full(len(ns), np.nan)
    low, high = [], []
    boundaries = np.r_[0, np.flatnonzero(np.diff(days)) + 1, len(ns)]
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if low:
            out[a:b] = -low[0] if len(low) > len(high) else (-low[0] + high[0]) / 2.0
        for j in range(a, b):
            value = sigma[j]
            if ns[j] < lower or not np.isfinite(value):
                continue
            if not low or value <= -low[0]:
                heapq.heappush(low, -float(value))
            else:
                heapq.heappush(high, float(value))
            if len(low) > len(high) + 1:
                heapq.heappush(high, -heapq.heappop(low))
            elif len(high) > len(low):
                heapq.heappush(low, -heapq.heappop(high))
    return pd.Series(out, index=index, name="causal_sigma_ref")


def causal_z_context(ctx: F.Context) -> F.Context:
    reference = causal_sigma_reference(ctx).to_numpy()
    sigma = ctx.frame["sigma7d"].to_numpy(dtype=float)
    scale = np.full(len(reference), np.nan)
    np.divide(sigma, reference, out=scale, where=np.isfinite(reference) & (reference > 0))
    cond = {"pct": {k: a.copy() for k, a in ctx.cond["pct"].items()}, "z": S.conditions(ctx.frame, scale)}
    return replace(ctx, cond=cond)


def fresh_context(ctx: F.Context, *, main_start=None, box_ok=None) -> F.Context:
    """Reconstruct with make_context; no source array/frame/dict is written."""
    start_i, last_i, _ = E._window_bounds(ctx)
    grid = pd.DataFrame({"open": ctx.O, "high": ctx.H, "low": ctx.L, "close": ctx.C,
                         "volume": ctx.V, "bad": ctx.bad}, index=ctx.index, copy=True)
    if ctx.source == F.SYNTHETIC_TAG:
        grid.attrs["provenance"] = F.SYNTHETIC_TAG
    fm = np.isfinite(ctx.funding_rate_open)
    funding = pd.Series(ctx.funding_rate_open[fm].copy(), index=ctx.index[fm]) if ctx.funding_supplied else None
    return F.make_context(grid, ctx.box_ok if box_ok is None else box_ok, ctx.source,
                          main_start=ctx.index[start_i] if main_start is None else main_start,
                          last_trigger=ctx.index[last_i], archive_gap_minutes=ctx.index[ctx.archive_gap], funding=funding)


def full_period_context(ctx: F.Context) -> F.Context:
    return fresh_context(ctx, main_start=IS_START)


def box_contexts(ctx: F.Context, df1m, spot_1d, spot_1m):
    """Yield ((variant, version), new_context), using the exact CP1 hookup.

    A generator avoids retaining forty complete copies of a 2.1M-minute context.
    All three datasets are explicitly supplied by the future CP4 runner.
    """
    from . import daily as DY
    from tools.run_impulse_cp1 import VARIANTS, STATE_VARIANTS, box_mask
    variants = tuple(v for v in VARIANTS if v != VS.BOX_STATE_VARIANT)
    state_variants = tuple(v for v in STATE_VARIANTS if v != VS.BOX_STATE_VARIANT)
    states = DY.build_box_states(df1m, spot_1d, spot_1m, variants=state_variants)
    c5, c15 = pd.Series(ctx.C).shift(5).to_numpy(), pd.Series(ctx.C).shift(15).to_numpy()
    minute_ns = pd.DatetimeIndex(ctx.index).tz_convert("UTC").as_unit("ns").asi8
    for variant in variants:
        for key in BOX_VERSIONS:
            spec = VS.VERSIONS[key]
            direction = "dn" if spec.direction == "dn_long" else "up"
            mask = box_mask(variant, ctx.cond[spec.trigger_rule], direction, minute_ns,
                            ctx.C, c15, c5, states)
            yield (variant, key), fresh_context(ctx, box_ok=mask)


@njit(cache=True)
def _variant_trade_path(i, s, sl, O, H, L, C, V, bad, fund_rate_open, base_med_i, fixed1, v1_min, k1_min, n, natural, optimistic):
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
    first_end = ((e // 15) + 1) * 15 - 1 if natural else i + 15
    second_end = first_end + 15
    if natural:
        for before in range((e // 15) * 15, e):
            if not bad[before]:
                h1 = max(h1, H[before])
                l1 = min(l1, L[before])
                vs += V[before] if V[before] == V[before] else 0.0
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
        prefer_close = False
        if optimistic and j == second_end:
            prefer_close = True
        elif optimistic and j == first_end:
            hv, lv = max(h1, H[j]), min(l1, L[j])
            vv = vs + (V[j] if V[j] == V[j] else 0.0)
            passes = (base_med_i > 0.0 and hv > lv and vv / base_med_i >= v1_min
                      and s * (2.0 * C[j] - hv - lv) / (hv - lv) >= k1_min)
            prefer_close = fixed1 or not passes
        if ((s > 0 and L[j] <= sl) or (s < 0 and H[j] >= sl)) and not prefer_close:    # 盘中止损（优先于同一分钟的收盘离场）
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
        if j == first_end:                                              # 第 1 根合成棒满：延续检验（B4 也检验，只作诊断）
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
        if j == second_end:
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
def _variant_portfolio(entry_dir, sl_arr, eref_arr, O, H, L, C, V, bad, fund_rate_open, base_med, start_i, last_i, end_i,
                   fixed1, v1_min, k1_min, fee, slip_e, slip_x, use_funding, risk_mode, risk_frac, min_dist_frac, lev_cap,
                   notional_mult, c_rt, init_cap, cooldown, rec, natural, optimistic):
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
        x_bar, x_ref, reason, b_last, t_exit, v1, k1, cont, mae, mfe, nf, fund_b, fund_unit = _variant_trade_path(
            i, s, sl, O, H, L, C, V, bad, fund_rate_open, base_med[((i + 1) // 15) * 15 - 1] if natural else base_med[i], fixed1, v1_min, k1_min, n, natural, optimistic)
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



def run_path_variant(ctx: F.Context, vin: F.VersionInputs, cost: VS.CostScenario,
                     sizing: VS.Sizing, *, variant: str,
                     initial_capital: float = VS.INITIAL_CAPITAL) -> E.PortfolioResult:
    """Frozen accounting with one of the section-11 path changes (or reference)."""
    if variant not in ("reference", "optimistic", "natural"):
        raise ValueError(f"unregistered path variant: {variant}")
    E._guard(ctx)
    start_i, last_i, end_i = E._window_bounds(ctx)
    E._check_version_inputs(ctx, vin)
    if not np.isin(vin.entry_dir, (-1, 0, 1)).all():
        raise ValueError("entry_dir must be -1, 0 or +1")
    sig = np.flatnonzero((vin.entry_dir != 0) & ctx.window)
    E._check_signals(ctx, sig, vin.entry_dir[sig].astype(float), vin.sl[sig], vin.eref[sig])
    if variant == "natural" and len(sig):
        previous_bar_end = ((sig + 1) // 15) * 15 - 1
        if (previous_bar_end < 299).any() or not np.isfinite(ctx.base_med[previous_bar_end]).all():
            raise ValueError("natural holding bars require twenty complete preceding natural bars")
    rec = np.full((len(sig) + 1, len(E.REC_COLS)), np.nan)
    ntr, equity = _variant_portfolio(
        vin.entry_dir, vin.sl, vin.eref, ctx.O, ctx.H, ctx.L, ctx.C, ctx.V, ctx.bad,
        ctx.funding_rate_open, ctx.base_med, start_i, last_i, end_i, vin.spec.hold == "fixed1",
        VS.V1_MIN, VS.K1_MIN, cost.fee_taker, cost.slip_entry, cost.slip_exit, cost.include_funding,
        sizing.mode == "risk", sizing.risk_fraction, sizing.min_dist_frac, sizing.leverage_cap,
        sizing.notional_mult, cost.c_rt, float(initial_capital), VS.COOLDOWN_MIN, rec,
        variant == "natural", variant == "optimistic")
    trades = pd.DataFrame(rec[:ntr], columns=E.REC_COLS)
    for col in E._INT_COLS:
        trades[col] = trades[col].astype(np.int64)
    for col in ("cap_bound", "floor_bound"):
        trades[col] = trades[col].astype(bool)
    trades = E._enrich(ctx, vin, trades)
    equity = pd.Series(equity, index=ctx.index, name="equity")
    valid = equity.dropna()
    daily = valid.groupby(valid.index.floor("D")).last()
    return E.PortfolioResult(trades, equity, daily, {
        "version": vin.spec.name, "cost": cost.name, "sizing": sizing.name, "path_variant": variant,
        "start_idx": start_i, "last_trigger_idx": last_i, "n_trades": ntr,
        "n_signals_in_window": int(len(sig)), "final_equity": float(equity.iloc[-1]),
        "eligibility_exclusions": vin.skipped,
        "eligibility_exclusions_exclusive": vin.skipped_exclusive})


def _continuation(ctx: F.Context, trigger: int, side: int) -> tuple[float, float, bool]:
    e, x = trigger + 1, trigger + 15
    high, low = float(np.max(ctx.H[e:x + 1])), float(np.min(ctx.L[e:x + 1]))
    volume = float(np.nansum(ctx.V[e:x + 1]))
    baseline = float(ctx.base_med[trigger])
    v1 = volume / baseline if baseline > 0 else np.nan
    k1 = side * (2.0 * ctx.C[x] - high - low) / (high - low) if high > low else np.nan
    return v1, k1, bool(v1 >= VS.V1_MIN and k1 >= VS.K1_MIN)


def classify_path_ambiguities(ctx: F.Context, vin: F.VersionInputs, trades: pd.DataFrame,
                             cost: VS.CostScenario = VS.COSTS[VS.MAIN_COST]) -> dict:
    """Section 11(2); recompute first-bar diagnostics even when cont == -1."""
    out = trades.copy(deep=True)
    conflict, economic, v1s, k1s = [], [], [], []
    for row in out.itertuples():
        i, x, s = int(row.trigger_idx), int(row.exit_bar), int(row.side)
        v1, k1 = row.v1, row.k1
        hit = False
        if row.reason == STOP and x == i + 15:
            v1, k1, cont = _continuation(ctx, i, s)
            hit = vin.spec.hold == "fixed1" or not cont
        elif row.reason == STOP and x == i + 30:
            hit = True
        delta = 0.0
        if hit:
            difference = (ctx.C[x] - row.exit_ref) * (1 - s * cost.slip_exit)
            delta = (s - cost.fee_taker) * difference
            if cost.include_funding and x + 1 < len(ctx.C) and np.isfinite(ctx.funding_rate_open[x + 1]):
                delta -= s * ctx.C[x] * ctx.funding_rate_open[x + 1]
        conflict.append(hit)
        economic.append(bool(hit and delta != 0.0))
        v1s.append(v1)
        k1s.append(k1)
    out["stop_close_conflict"] = np.asarray(conflict, dtype=bool)
    out["economic_result_changes"] = np.asarray(economic, dtype=bool)
    out["diagnostic_v1"] = np.asarray(v1s, dtype=float)
    out["diagnostic_k1"] = np.asarray(k1s, dtype=float)
    out["entry_open_gap"] = out["reason"] == GAP_ENTRY
    out["holding_open_gap"] = out["reason"] == GAP_STOP
    out["entry_minute_intrabar_stop"] = (out["reason"] == STOP) & (out["entry_idx"] == out["exit_bar"])
    columns = ("stop_close_conflict", "economic_result_changes", "entry_open_gap", "holding_open_gap",
               "entry_minute_intrabar_stop")
    counts = {key: {"n": int(out[key].sum()), "share": float(out[key].mean()) if len(out) else None}
              for key in columns}
    return {"counts": counts, "trades": out, "archive_gap_trades": out.loc[out["has_archive_gap"]].copy()}


def reconstruct_equity(ctx: F.Context, trades: pd.DataFrame, *, include_funding: bool,
                       initial_capital: float = VS.INITIAL_CAPITAL) -> tuple[pd.Series, pd.Series]:
    """Independent cash-event reconstruction from ledger quantities/fills and marks.

    Cash fees and realized fill P&L are read from ledger fields. Each funding
    amount and posting minute is independently recomputed, including the special
    previous-day attribution of a midnight time exit. No engine path/accounting
    helper is used. Ledger totals are checked separately from equity comparisons.
    """
    start_i, _, end_i = E._window_bounds(ctx)
    cash_delta = np.zeros(len(ctx.C))
    marked = np.zeros(len(ctx.C))
    last_exit = -1
    for row in trades.sort_values("entry_idx", kind="stable").itertuples():
        e, x, side, qty = int(row.entry_idx), int(row.exit_bar), int(row.side), float(row.qty)
        if e <= last_exit or e < start_i or x < e or x > end_i:
            raise ValueError("ledger entries overlap or leave the reconstruction window")
        last_exit = x
        cash_delta[e] -= float(row.fee_entry)
        cash_delta[x] += side * qty * (float(row.exit_fill) - float(row.entry_fill)) - float(row.fee_exit)
        funding = 0.0
        settlements = 0
        last_mark = float(ctx.C[int(row.trigger_idx)])
        for minute in range(e, x + 1):
            if minute > e and np.isfinite(ctx.funding_rate_open[minute]):
                price = float(ctx.O[minute]) if np.isfinite(ctx.O[minute]) else last_mark
                unit = -side * price * float(ctx.funding_rate_open[minute])
                settlements += 1
                if include_funding:
                    cash_delta[minute] += qty * unit
                    funding += qty * unit
            if minute < x:
                price = float(ctx.C[minute]) if np.isfinite(ctx.C[minute]) else last_mark
                marked[minute] = side * qty * (price - float(row.entry_fill))
                last_mark = price
        if row.reason in (TIME1, TIME2) and x + 1 < len(ctx.C) and np.isfinite(ctx.funding_rate_open[x + 1]):
            settlements += 1
            if include_funding:
                amount = -side * qty * float(ctx.C[x]) * float(ctx.funding_rate_open[x + 1])
                cash_delta[x] += amount
                funding += amount
        if settlements != int(row.n_funding) or not np.isclose(funding, row.funding, rtol=1e-12, atol=1e-9):
            raise AssertionError("independent funding reconstruction disagrees with trade ledger")
        net = side * qty * (row.exit_fill - row.entry_fill) - row.fee_entry - row.fee_exit + funding
        if not np.isclose(net, row.net_pnl, rtol=1e-12, atol=1e-9):
            raise AssertionError("independent fill accounting disagrees with trade net P&L")
    values = initial_capital + np.cumsum(cash_delta) + marked
    values[:start_i] = np.nan
    equity = pd.Series(values, index=ctx.index, name="equity")
    valid = equity.dropna()
    daily = valid.groupby(valid.index.floor("D")).last()
    return equity, daily


def reconcile_portfolio(ctx: F.Context, vin: F.VersionInputs, result: E.PortfolioResult,
                        cost: VS.CostScenario, initial_capital: float = VS.INITIAL_CAPITAL) -> dict:
    """Verify all four section-11(1) identities, including independent daily equity."""
    E._check_version_inputs(ctx, vin)
    c = ctx.cond[vin.spec.trigger_rule]
    up, dn = c["up1"] | c["up2"], c["dn1"] | c["dn2"]
    triggered = up | dn if vin.spec.direction == "bi" else (dn if vin.spec.direction == "dn_long" else up)
    opportunities = int((triggered & ctx.window).sum())
    signals = int(((vin.entry_dir != 0) & ctx.window).sum())
    excluded = sum(vin.skipped_exclusive.values())
    traded = result.trades["trigger_idx"].to_numpy(dtype=int)
    if opportunities != signals + excluded:
        raise AssertionError("trigger minutes != signals + exclusive eligibility exclusions")
    if len(np.unique(traded)) != len(traded) or not (vin.entry_dir[traded] != 0).all():
        raise AssertionError("ledger contains duplicate trades or entries without a valid signal")
    accepted = np.zeros(len(ctx.C), dtype=bool)
    accepted[traded] = True
    skipped = ((vin.entry_dir != 0) & ctx.window) & ~accepted
    unavailable = np.zeros(len(ctx.C), dtype=bool)
    for row in result.trades.itertuples():
        unavailable[int(row.trigger_idx) + 1:int(row.exit_time_idx) + VS.COOLDOWN_MIN - 1] = True
    if not unavailable[skipped].all() or signals != len(traded) + int(skipped.sum()):
        raise AssertionError("signals do not reconcile to trades and holding/cooldown skips")
    net = float(result.trades["net_pnl"].sum())
    equity_change = float(result.equity.iloc[-1]) - initial_capital
    if not np.isclose(net, equity_change, rtol=1e-12, atol=1e-7):
        raise AssertionError("ledger net P&L != final equity - initial capital")
    equity, daily = reconstruct_equity(ctx, result.trades, include_funding=cost.include_funding,
                                       initial_capital=initial_capital)
    if not daily.index.equals(result.daily_equity.index):
        raise AssertionError("independent daily equity index mismatch")
    if not np.allclose(daily, result.daily_equity, rtol=1e-12, atol=1e-7):
        raise AssertionError("independent daily equity mismatch")
    if not np.allclose(equity, result.equity, rtol=1e-12, atol=1e-7, equal_nan=True):
        raise AssertionError("independent minute equity mismatch")
    return {"trigger_minutes": opportunities, "signals": signals, "exclusive_exclusions": excluded,
            "trades": len(traded), "holding_or_cooldown_skips": int(skipped.sum()),
            "ledger_net_pnl": net, "final_equity_change": equity_change,
            "max_daily_equity_difference": float(np.max(np.abs(daily - result.daily_equity))),
            "all_identities_passed": True}


def defect_filter(ctx: F.Context, result: E.PortfolioResult, cost: VS.CostScenario) -> dict:
    mask = result.trades.loc[:, list(DEFECT_FLAGS)].any(axis=1)
    clean = result.trades.loc[~mask].copy(deep=True)
    equity, daily = reconstruct_equity(ctx, clean, include_funding=cost.include_funding)
    meta = {**result.meta, "sensitivity": "exclude_any_data_defect", "n_trades": len(clean),
            "final_equity": float(equity.iloc[-1]), "retained_original_quantities": True}
    return {"result": E.PortfolioResult(clean, equity, daily, meta),
            "excluded": result.trades.loc[mask].copy(deep=True),
            "retained_trigger_indices": clean["trigger_idx"].to_numpy(dtype=np.int64)}


def _run_versions(ctx: F.Context, names, *, variant=None) -> dict:
    results = {}
    for name in names:
        vin = F.build_version_inputs(ctx, VS.VERSIONS[name])
        if variant is None:
            results[name] = E.run_portfolio(ctx, vin, VS.COSTS[VS.MAIN_COST], VS.SIZINGS[VS.MAIN_SIZING])
        else:
            results[name] = run_path_variant(ctx, vin, VS.COSTS[VS.MAIN_COST],
                                            VS.SIZINGS[VS.MAIN_SIZING], variant=variant)
    return results


def run_cp3_sensitivities(ctx: F.Context) -> dict:
    """Only G8z and section 11(9), called by the authorized once-only CP3 runner."""
    causal = causal_z_context(ctx)
    causal_results = _run_versions(causal, Z_VERSIONS)
    full = full_period_context(ctx)
    full_results = _run_versions(full, B_VERSIONS)
    return {"causal_context": causal, "causal_results": causal_results,
            "full_period_context": full, "full_period_results": full_results,
            "causal_sigma_reference": causal_sigma_reference(ctx),
            "appendix_enters_gates": False}


def run_cp4_sensitivities(ctx: F.Context,
                         all_results: Mapping[tuple[str, str, str], E.PortfolioResult], *,
                         cp3_results: dict, box_data: tuple,
                         statistics_hook: Callable | None = None) -> dict:
    """Explicit future CP4 orchestration of section 11(1--9), never called by CP3.

    Each hook receives ONE dict with section, label, seed_shift, context,
    results, and applicable auxiliary fields. It supplies the frozen statistics
    (including retained matched excess rows for section 3). The hook must not
    promote sensitivity output into the primary decision. Section 8 uses its
    dedicated seed rows (seed_shift=0), not the k=2..7 sensitivity shift.
    """
    main = {name: all_results[name, VS.MAIN_COST, VS.MAIN_SIZING] for name in VS.VERSIONS}
    harsh = {name: all_results[name, "harsh_34bp", VS.MAIN_SIZING] for name in VS.VERSIONS}
    out = {}

    def emit(section, label, results, **extra):
        payload = {"section": section, "label": label,
                   "seed_shift": 10000 * section if 2 <= section <= 7 else 0,
                   "context": ctx, "results": results, "enters_primary_gates": False, **extra}
        if statistics_hook is not None:
            payload["statistics"] = statistics_hook(payload)
        out[label] = payload

    reconciliations, paths = {}, {}
    for name, result in main.items():
        vin = F.build_version_inputs(ctx, VS.VERSIONS[name])
        reconciliations[name] = reconcile_portfolio(ctx, vin, result, VS.COSTS[VS.MAIN_COST])
        paths[name] = classify_path_ambiguities(ctx, vin, result.trades)
    emit(1, "reconciliation", main, reconciliation=reconciliations)
    emit(2, "optimistic_path", _run_versions(ctx, VS.VERSIONS, variant="optimistic"), path_diagnostics=paths)
    clean = {name: defect_filter(ctx, result, VS.COSTS[VS.MAIN_COST]) for name, result in main.items()}
    clean_harsh = {name: defect_filter(ctx, result, VS.COSTS["harsh_34bp"]) for name, result in harsh.items()}
    emit(3, "exclude_data_defects", {name: item["result"] for name, item in clean.items()},
         harsh_results={name: item["result"] for name, item in clean_harsh.items()},
         excluded={name: item["excluded"] for name, item in clean.items()},
         retained_trigger_indices={name: item["retained_trigger_indices"] for name, item in clean.items()})
    emit(4, "natural_quarter_hour", _run_versions(ctx, VS.VERSIONS, variant="natural"))
    comparisons = {name: main[name].trades.loc[main[name].trades["trigger_time"] >= FIRST_Z].copy()
                   for name in ("A2_box_origin", "A5_box_origin_z", "B3_pure_bi_mom", "B5_pure_bi_mom_z")}
    emit(5, "causal_sigma_reference", cp3_results["causal_results"],
         context=cp3_results["causal_context"], pct_z_comparison_trades=comparisons)
    emit(6, "costs_and_sizing", dict(all_results))
    for (variant, name), alternate in box_contexts(ctx, *box_data):
        results = _run_versions(alternate, (name,))
        emit(7, f"box_{variant}_{name}", results, context=alternate, box_variant=variant)
        # All bars already live in the shared caller; avoid keeping a new full
        # context for every variant after its statistics hook has finished.
        out[f"box_{variant}_{name}"]["context"] = None
    emit(8, "daily_clusters_and_stationary_bootstrap", main, harsh_results=harsh,
         cluster="day", sharpe_auxiliary="stationary_mean_block_10_days")
    emit(9, "b_full_period_appendix", cp3_results["full_period_results"],
         context=cp3_results["full_period_context"], matched_controls=False)
    return out

"""
impulse_study/features.py — Checkpoint 2：引擎所需的因果特征与 10 个版本的逐分钟入场输入（只用触发时刻及之前的数据）。

三种“15 分钟”严格分开（实验计划第 4 节）：
1. 脉冲收益 r15 = C_T / C_{T−15} − 1（滚动，signals.py）；
2. ATR_15m(14)：UTC 自然整刻（00/15/30/45）已收盘 15m 棒，只用有效（~bad）1m 聚合 —— 本模块 ``atr15_natural``；
3. 持有期合成 15m 棒：以入场时刻为锚点每 15 根 1m（engine.py），绝不按自然整刻对齐。

ATR_15m(14) 规则
----------------
* 桶 b 覆盖 [b·15min, (b+1)·15min)（origin = epoch，label = left，closed = left），只聚合 ~bad 的 1m：n_valid ∈ 0..15，
  O = 第一根有效开盘，H / L = 有效最高 / 最低，C = 最后一根有效收盘。
* 1 ≤ n_valid ≤ 15：TR = max(H − L, |H − prev_valid_close|, |L − prev_valid_close|)（史上第一根有效桶 TR = H − L），
  之后 prev_valid_close = C。
* n_valid = 0：不计 TR、ATR 不衰减（沿用上一有效值），prev_valid_close 不变。
* 前 14 个有效 TR 的算术均值为首个 ATR，其后 ATR_k = (13·ATR_{k−1} + TR_k) / 14；初始化前为 NaN（不得交易）。
* 读取时点：触发 K 线 T 在 T + 1min 收盘决策，只读 close_time ≤ T + 1min 的最近一根已收盘桶。

基准成交量（第 1 根合成棒的放量比 v1 的分母）：入场前紧邻的 20 个不重叠 15 分钟窗口 [T − 299min, T + 1min)，
每个窗口的有效成交量（缺失记 0，维护占位本身成交量为 0）之和，取 20 个窗口和的中位数（偶数个取中间两个的平均）。

脉冲起点极值（SL_B / SL_C）：多头 L_origin = min(C_{t−W}, min L_{t−W+1..t})；空头 H_origin = max(C_{t−W}, max H_{t−W+1..t})。
W 只看该版本自身阈值口径、该笔入场方向的指标：该方向指标 ② 成立（含 ① + ②）→ W = 15；仅指标 ① → W = 5。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from numba_compat import njit

from . import signals as S
from . import versions as VS
from .scope import MINUTE_NS

FIFTEEN = 15


# ----------------------------------------------------------------------------
# ATR_15m(14)：自然整刻 15m 棒
# ----------------------------------------------------------------------------
@njit(cache=True)
def _buckets_15m(O, H, L, C, bad, nb):
    n_valid = np.zeros(nb, dtype=np.int64)
    bo = np.full(nb, np.nan)
    bh = np.full(nb, np.nan)
    bl = np.full(nb, np.nan)
    bc = np.full(nb, np.nan)
    for b in range(nb):
        for k in range(FIFTEEN):
            i = b * FIFTEEN + k
            if bad[i]:
                continue
            if n_valid[b] == 0:
                bo[b] = O[i]
                bh[b] = H[i]
                bl[b] = L[i]
            else:
                if H[i] > bh[b]:
                    bh[b] = H[i]
                if L[i] < bl[b]:
                    bl[b] = L[i]
            bc[b] = C[i]
            n_valid[b] += 1
    return n_valid, bo, bh, bl, bc


@njit(cache=True)
def _atr_buckets(n_valid, bh, bl, bc, n_len):
    nb = n_valid.shape[0]
    atr = np.full(nb, np.nan)
    tr_out = np.full(nb, np.nan)
    prev_close = np.nan
    cur = np.nan
    acc = 0.0
    n_tr = 0
    for b in range(nb):
        if n_valid[b] == 0:
            atr[b] = cur                      # 整根维护缺失：不计 TR、不衰减，沿用上一有效值（可能仍为 NaN）
            continue
        hl = bh[b] - bl[b]
        if prev_close != prev_close:          # 史上第一根有效桶
            tr = hl
        else:
            tr = max(hl, max(abs(bh[b] - prev_close), abs(bl[b] - prev_close)))
        tr_out[b] = tr
        n_tr += 1
        if n_tr <= n_len:
            acc += tr
            if n_tr == n_len:
                cur = acc / n_len
        else:
            cur = ((n_len - 1) * cur + tr) / n_len
        atr[b] = cur
        prev_close = bc[b]
    return atr, tr_out


def atr15_natural(O, H, L, C, bad, start_ns: int) -> dict:
    """返回桶表与逐分钟决策时可读的 ATR（atr_at[t] = close_time ≤ T + 1min 的最近一根已收盘桶的 ATR）。"""
    if start_ns % (FIFTEEN * MINUTE_NS) != 0:
        raise ValueError("minute grid must start on a UTC 15-minute boundary")
    n = len(C)
    nb = n // FIFTEEN
    bad_b = np.asarray(bad, dtype=np.bool_)
    n_valid, bo, bh, bl, bc = _buckets_15m(np.asarray(O, float), np.asarray(H, float), np.asarray(L, float),
                                           np.asarray(C, float), bad_b, nb)
    atr, tr = _atr_buckets(n_valid, bh, bl, bc, VS.ATR15_LEN)
    t = np.arange(n)
    b_last = (t + 1) // FIFTEEN - 1               # 决策时刻 T + 1min 已收盘的最近一根桶
    ok = (b_last >= 0) & (b_last < nb)
    atr_at = np.full(n, np.nan)
    atr_at[ok] = atr[b_last[ok]]
    return {"n_valid": n_valid, "open": bo, "high": bh, "low": bl, "close": bc, "tr": tr, "atr": atr,
            "atr_at": atr_at, "bucket_start_ns": start_ns + np.arange(nb, dtype=np.int64) * FIFTEEN * MINUTE_NS}


# ----------------------------------------------------------------------------
# 基准成交量中位数
# ----------------------------------------------------------------------------
@njit(cache=True)
def _baseline_median(vol, n_win, w):
    n = vol.shape[0]
    cs = np.zeros(n + 1)
    for i in range(n):
        v = vol[i]
        cs[i + 1] = cs[i] + (v if v == v else 0.0)
    out = np.full(n, np.nan)
    buf = np.empty(n_win)
    for t in range(n_win * w - 1, n):
        for k in range(n_win):
            hi = t + 1 - k * w                    # 第 k+1 个窗口 = 分钟 [hi − w, hi)
            buf[k] = cs[hi] - cs[hi - w]
        srt = np.sort(buf)
        if n_win % 2 == 1:
            out[t] = srt[n_win // 2]
        else:
            out[t] = 0.5 * (srt[n_win // 2 - 1] + srt[n_win // 2])
    return out


def baseline_median(volume, bad) -> np.ndarray:
    v = np.where(np.asarray(bad, dtype=bool), 0.0, np.nan_to_num(np.asarray(volume, float), nan=0.0))
    return _baseline_median(v, VS.BASELINE_WINDOWS, VS.BAR_MIN)


# ----------------------------------------------------------------------------
# 脉冲起点极值
# ----------------------------------------------------------------------------
def origin_extremes(close, low, high, w: int) -> tuple[np.ndarray, np.ndarray]:
    """L_origin(t) = min(C_{t−W}, min L_{t−W+1..t})；H_origin(t) = max(C_{t−W}, max H_{t−W+1..t})。"""
    cw = pd.Series(np.asarray(close, float)).shift(w).to_numpy()
    lo = pd.Series(np.asarray(low, float)).rolling(w, min_periods=w).min().to_numpy()
    hi = pd.Series(np.asarray(high, float)).rolling(w, min_periods=w).max().to_numpy()
    return np.minimum(cw, lo), np.maximum(cw, hi)                # 任一项缺失则为 NaN


# ----------------------------------------------------------------------------
# 上下文与版本输入
# ----------------------------------------------------------------------------
@dataclass
class Context:
    """一份分钟网格上的全部因果特征（真实或合成数据）。source = 'synthetic' / 'binance_is'。"""
    index: pd.DatetimeIndex
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray
    bad: np.ndarray
    frame: pd.DataFrame          # signals.impulse_frame 的输出（r1/r5/r15/σ7d/contaminated…）
    cond: dict                   # {'pct': {...}, 'z': {...}}
    box_ok: np.ndarray           # S6 二段支撑（按 C_T 判定）
    atr: dict
    base_med: np.ndarray
    window: np.ndarray           # MAIN_START ≤ T ≤ LAST_TRIGGER
    archive_gap: np.ndarray      # CP1 表 5b 的归档末尾漏记分钟
    funding_rate_open: np.ndarray  # 开盘时刻恰为资金费结算时点的分钟 → 费率（否则 NaN）
    source: str


def make_context(grid: pd.DataFrame, box_ok: np.ndarray, source: str, main_start=VS.MAIN_START,
                 last_trigger=VS.LAST_TRIGGER, archive_gap_minutes=(), funding: pd.Series | None = None) -> Context:
    """grid = data_1m.minute_grid(...)（完整分钟网格 + bad 标记）。funding：以结算时刻为索引的费率序列。"""
    idx = pd.DatetimeIndex(grid.index).as_unit("ns")
    ns = np.asarray(idx.asi8)
    O, H, L, C, V = (grid[c].to_numpy(dtype=float, copy=True) for c in ("open", "high", "low", "close", "volume"))
    bad = grid["bad"].to_numpy(dtype=bool, copy=True)
    f = S.impulse_frame(grid)
    cond = {"pct": S.conditions(f), "z": S.conditions(f, (f["sigma7d"] / VS.SIGMA_REF).to_numpy())}
    atr = atr15_natural(O, H, L, C, bad, int(ns[0]))
    base = baseline_median(V, bad)
    window = (ns >= int(pd.Timestamp(main_start).value)) & (ns <= int(pd.Timestamp(last_trigger).value))
    gap = np.zeros(len(ns), dtype=bool)
    if len(archive_gap_minutes):
        gap = np.isin(ns, np.asarray(pd.DatetimeIndex(archive_gap_minutes).as_unit("ns").asi8))
    fr = np.full(len(ns), np.nan)
    if funding is not None and len(funding):
        fi = np.asarray(pd.DatetimeIndex(funding.index).as_unit("ns").asi8)
        pos = np.searchsorted(ns, fi)
        ok = (pos < len(ns)) & (ns[np.minimum(pos, len(ns) - 1)] == fi)
        if (~ok & (fi >= ns[0]) & (fi <= ns[-1])).any():
            raise ValueError("funding timestamps fall off the minute grid")
        fr[pos[ok]] = np.asarray(funding.to_numpy(dtype=float))[ok]
    return Context(idx, O, H, L, C, V, bad, f, cond, np.asarray(box_ok, dtype=bool), atr, base, window, gap, fr, source)


@dataclass
class VersionInputs:
    spec: VS.VersionSpec
    entry_dir: np.ndarray        # int8：+1 做多、−1 做空、0 无
    sl: np.ndarray               # 冻结的止损价（无信号处 NaN）
    eref: np.ndarray             # 下单参考价 E_ref = C_t
    w: np.ndarray                # int8：止损回溯窗口（SL_A 为 0）
    cat: np.ndarray              # int8：入场方向的触发类别 1 仅① / 2 仅② / 3 ①+②
    atr_at: np.ndarray
    skipped: dict                # 被各项资格条件排除的触发计数（审计用）


def build_version_inputs(ctx: Context, spec: VS.VersionSpec) -> VersionInputs:
    c = ctx.cond[spec.trigger_rule]
    n = len(ctx.C)
    up = c["up1"] | c["up2"]
    dn = c["dn1"] | c["dn2"]
    atr_at = ctx.atr["atr_at"]
    entry_ok = np.zeros(n, dtype=bool)
    entry_ok[:-1] = ~ctx.bad[1:]                       # 入场 K 线 t+1 必须是有效分钟（否则信号作废）
    contam = ctx.frame["contaminated"].to_numpy()
    eligible = ctx.window & ~contam & np.isfinite(atr_at) & entry_ok & np.isfinite(ctx.base_med)
    if spec.box:
        eligible = eligible & ctx.box_ok
    d = np.zeros(n, dtype=np.int8)
    cat = np.zeros(n, dtype=np.int8)
    if spec.direction == "up_long":
        m = up & eligible
        d[m] = 1
        cat[m] = np.where(c["up1"] & c["up2"], 3, np.where(c["up2"], 2, 1))[m]
        ind2 = c["up2"]
    elif spec.direction == "dn_long":
        m = dn & eligible
        d[m] = 1
        cat[m] = np.where(c["dn1"] & c["dn2"], 3, np.where(c["dn2"], 2, 1))[m]
        ind2 = c["dn2"]
    elif spec.direction == "bi":
        if not spec.ambiguous_skip:
            raise ValueError("bidirectional versions must skip ambiguous minutes")
        amb = up & dn
        mu = up & eligible & ~amb
        md = dn & eligible & ~amb
        d[mu] = 1
        d[md] = -1
        cat[mu] = np.where(c["up1"] & c["up2"], 3, np.where(c["up2"], 2, 1))[mu]
        cat[md] = np.where(c["dn1"] & c["dn2"], 3, np.where(c["dn2"], 2, 1))[md]
        ind2 = np.where(d == 1, c["up2"], c["dn2"])
    else:
        raise ValueError(spec.direction)
    eref = ctx.C.copy()
    sl = np.full(n, np.nan)
    w = np.zeros(n, dtype=np.int8)
    s = d.astype(float)
    if spec.stop == "A":
        sl = np.where(d != 0, eref - s * VS.SL_A_MULT * atr_at, np.nan)
    else:
        w = np.where(d != 0, np.where(ind2, VS.W_IND2, VS.W_IND1), 0).astype(np.int8)
        lo5, hi5 = origin_extremes(ctx.C, ctx.L, ctx.H, VS.W_IND1)
        lo15, hi15 = origin_extremes(ctx.C, ctx.L, ctx.H, VS.W_IND2)
        lo = np.where(w == VS.W_IND2, lo15, lo5)
        hi = np.where(w == VS.W_IND2, hi15, hi5)
        if spec.stop == "B":
            sl = np.where(d == 1, lo - VS.SL_B_BUFFER * atr_at, np.where(d == -1, hi + VS.SL_B_BUFFER * atr_at, np.nan))
        elif spec.stop == "C":
            sl = np.where(d == 1, eref - VS.SL_C_FRAC * (eref - lo),
                          np.where(d == -1, eref + VS.SL_C_FRAC * (hi - eref), np.nan))
        else:
            raise ValueError(spec.stop)
    bad_sl = (d != 0) & ~(np.isfinite(sl) & (s * (eref - sl) > 0))
    skipped = {"invalid_stop": int(bad_sl.sum())}
    d[bad_sl] = 0
    trig_any = (up | dn) if spec.direction == "bi" else (up if spec.direction == "up_long" else dn)
    w_mask = ctx.window & trig_any
    skipped.update({"contaminated": int((w_mask & contam).sum()), "atr_not_ready": int((w_mask & ~np.isfinite(atr_at)).sum()),
                    "entry_bar_bad": int((w_mask & ~entry_ok).sum()),
                    "ambiguous": int((w_mask & up & dn).sum()) if spec.direction == "bi" else 0,
                    "box_rejected": int((w_mask & ~ctx.box_ok).sum()) if spec.box else 0})
    return VersionInputs(spec, d, np.where(d != 0, sl, np.nan), eref, np.where(d != 0, w, 0).astype(np.int8),
                         np.where(d != 0, cat, 0).astype(np.int8), atr_at, skipped)

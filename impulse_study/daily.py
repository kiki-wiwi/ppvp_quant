"""
impulse_study/daily.py — A 组“日线箱体二段反弹”背景：UTC 日线、ATR_1D、滚动 60 日成交量分布与二段支撑状态（因果）。

定义（Checkpoint 1 提议；参数与口径在 Checkpoint 2 的 experiment_plan_impulse.md 中冻结）
-------------------------------------------------------------------------------------------
1. UTC 日线 d 由 1m 按 UTC 00:00 边界聚合；``available_time(d) = open_time(d) + 24h``（该日完整收盘的时刻）。
   数据末端未走完的日线被剔除（available_time 晚于最后一根 1m 的收盘）。
2. 决策分钟 T（1m K 线开盘时间，该 K 线在 T + 1min 收盘，触发后于 T + 1min 开盘入场）只使用
   ``available_time ≤ T`` 的日线，即 ``as_of(T) = floor_day(T)``，窗口为 [as_of − 60d, as_of − 1d] 的 60 根已收盘日线。
   这比因果性要求多保守 1 分钟（23:59 的 K 线在 00:00 收盘时前一日恰好收盘，但不使用它），与仓库
   “1H K 线只使用 available_time ≤ open_time 的箱体”的约定一致。
3. 成交量分布（主口径 source='1m'）：窗口内全部 1m K 线的成交量在其 [low, high] 内均匀分配到 50 个等宽价格桶
   （桶边界 = np.linspace(窗口最低价, 窗口最高价, 51)），一字线计入价格所在桶；POC = 最大桶中心（1e-9 相对容差内并列取最低），
   价值区 70%（自 POC 向量较大一侧逐桶扩展，并列向上），VAL / VAH = 价值区下 / 上沿。分配与价值区算法与
   ppvp_engine.volume_profile / _value_area 相同（后者直接复用；前者为等价的逐根实现，测试中与原函数逐位比对）。
4. ATR_1D(14)：日线 Wilder ATR（ppvp_engine.wilder_atr，首值为前 14 根 TR 的简单平均），第 d 根在 d 收盘时可知，
   as_of 日使用 ATR[as_of − 1d]。价格预热：2019-09-01 起的现货日线（只用价格）与 2020-01-01 起的永续日线拼接。
5. 二段支撑状态（在 as_of 日内恒定）：band_lo = VAL − 0.5·ATR，band_hi = VAL + 0.5·ATR，bounce = band_hi + 0.8·ATR；
   过去 30 根已收盘日线 [as_of − 30d, as_of − 1d] 中存在 j1 < j2：日线 j1 触及支撑带（low ≤ band_hi 且 high ≥ band_lo），
   日线 j2 的最高价 ≥ bounce（完成第 1 段反弹）。VAL / ATR 取 as_of 当时的值（“当前”支撑带）。
6. 分钟级过滤：``box_ok(T) = primed(as_of) ∧ band_lo ≤ ref(T) ≤ POC``，主口径 ref(T) = 触发 K 线收盘价 C_T。
7. 预热：成交量分布只用永续成交量（不跨市场拼接），第一个完整 60 日窗口的 as_of 为 2020-03-01；此前 A 组不交易。

敏感性（只做计数，Checkpoint 1 不选择）：S1a 分布用 1H K 线、S1b 分布用日线；S2 允许 j1 = j2（同一根日线内先触及后反弹）；
S3 ref = 脉冲起点收盘价；S4 用 2019-11/12 现货 1m 成交量拼接预热（as_of 自 2020-01-01 起可用）；
S5 触及与反弹按各自当日（as_of = j1）的支撑带判定（时点口径），当前区间仍用当前值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from numba_compat import njit
from ppvp_engine import _value_area, wilder_atr

from .data_1m import agg_ohlcv
from .scope import DAY, DAY_NS, MINUTE_NS

W_PROFILE_DAYS = 60
W_TOUCH_DAYS = 30
N_BINS = 50
VA_PCT = 0.70
ATR_LEN_1D = 14
K_BAND = 0.5
K_BOUNCE = 0.8


# ----------------------------------------------------------------------------
# 日线
# ----------------------------------------------------------------------------
def daily_from_1m(df1m: pd.DataFrame) -> pd.DataFrame:
    """UTC 日线（只保留已完整收盘者），列 open/high/low/close/volume/n_min/available_time。"""
    d = agg_ohlcv(df1m, "1D")[["open", "high", "low", "close", "volume", "n_min"]].copy()
    d.index = pd.DatetimeIndex(d.index).as_unit("ns")
    d["available_time"] = d.index + DAY
    last_close = pd.DatetimeIndex(df1m.index).as_unit("ns")[-1] + pd.Timedelta(minutes=1)
    return d.loc[d["available_time"] <= last_close].copy()


def atr_1d(perp_daily: pd.DataFrame, spot_daily_warmup: pd.DataFrame | None, n: int = ATR_LEN_1D) -> pd.Series:
    """日线 Wilder ATR；有现货预热时先拼接 2020 年之前的现货日线（只用价格）。返回仅含永续日期的序列。"""
    cols = ["high", "low", "close"]
    if spot_daily_warmup is not None and len(spot_daily_warmup):
        w = spot_daily_warmup[spot_daily_warmup.index < perp_daily.index[0]][cols]
        x = pd.concat([w, perp_daily[cols]])
    else:
        x = perp_daily[cols]
    a = wilder_atr(x["high"].to_numpy(), x["low"].to_numpy(), x["close"].to_numpy(), n)
    return pd.Series(a, index=x.index).loc[perp_daily.index]


# ----------------------------------------------------------------------------
# 成交量分布（与 ppvp_engine.volume_profile 相同的均匀分配，逐根实现）
# ----------------------------------------------------------------------------
@njit(cache=True)
def _hist_uniform(lo, hi, vol, edges):
    nb = edges.shape[0] - 1
    p_low = edges[0]
    width = (edges[nb] - p_low) / nb
    hist = np.zeros(nb)
    for i in range(lo.shape[0]):
        L = lo[i]
        H = hi[i]
        v = vol[i]
        if H > L:
            rng = H - L
            b0 = int((L - p_low) / width) - 1
            b1 = int((H - p_low) / width) + 1
            if b0 < 0:
                b0 = 0
            if b1 > nb - 1:
                b1 = nb - 1
            for j in range(b0, b1 + 1):
                ov = min(H, edges[j + 1]) - max(L, edges[j])
                if ov > 0.0:
                    hist[j] += ov / rng * v
        else:
            k = int((L - p_low) / width)
            if k < 0:
                k = 0
            if k > nb - 1:
                k = nb - 1
            hist[k] += v
    return hist


def profile_levels(lo: np.ndarray, hi: np.ndarray, vol: np.ndarray, n_bins: int = N_BINS, va_pct: float = VA_PCT) -> tuple:
    """(POC, VAL, VAH, p_low, p_high)；与 ppvp_engine.volume_profile 的定义相同。"""
    lo = np.ascontiguousarray(lo, dtype=np.float64)
    hi = np.ascontiguousarray(hi, dtype=np.float64)
    vol = np.ascontiguousarray(vol, dtype=np.float64)
    p_low, p_high = float(lo.min()), float(hi.max())
    if not p_high > p_low:
        return p_low, p_low, p_high, p_low, p_high
    edges = np.linspace(p_low, p_high, n_bins + 1)
    hist = _hist_uniform(lo, hi, vol, edges)
    poc_i, lo_i, hi_i = _value_area(hist, float(va_pct))
    return 0.5 * (edges[poc_i] + edges[poc_i + 1]), float(edges[lo_i]), float(edges[hi_i + 1]), p_low, p_high


def rolling_profiles(bars: pd.DataFrame, as_of_days: pd.DatetimeIndex, window_days: int = W_PROFILE_DAYS,
                     n_bins: int = N_BINS, va_pct: float = VA_PCT, min_first_bar: pd.Timestamp | None = None) -> pd.DataFrame:
    """对每个 as_of 日，用开盘时间 ∈ [as_of − window, as_of) 的 K 线（任意周期，须已收盘）计算分布。

    窗口起点早于 min_first_bar（可用数据起点）的 as_of 记为 NaN（预热不足）。
    """
    t = np.asarray(pd.DatetimeIndex(bars.index).as_unit("ns").asi8, dtype=np.int64)
    lo, hi, v = (np.ascontiguousarray(bars[c].to_numpy(dtype=np.float64)) for c in ("low", "high", "volume"))
    first = int(pd.Timestamp(min_first_bar).as_unit("ns").value) if min_first_bar is not None else int(t[0])
    rows = []
    for D in pd.DatetimeIndex(as_of_days).as_unit("ns"):
        e_ns = int(D.value)
        s_ns = e_ns - window_days * DAY_NS
        if s_ns < first:
            rows.append((np.nan,) * 5 + (0,))
            continue
        s = int(np.searchsorted(t, s_ns, side="left"))
        e = int(np.searchsorted(t, e_ns, side="left"))
        rows.append(profile_levels(lo[s:e], hi[s:e], v[s:e], n_bins, va_pct) + (e - s,))
    out = pd.DataFrame(rows, columns=["POC", "VAL", "VAH", "P_low", "P_high", "n_bars"], index=pd.DatetimeIndex(as_of_days))
    out.index.name = "as_of"
    return out


# ----------------------------------------------------------------------------
# 二段支撑状态
# ----------------------------------------------------------------------------
def box_state(daily: pd.DataFrame, prof: pd.DataFrame, atr: pd.Series, touch_days: int = W_TOUCH_DAYS,
              k_band: float = K_BAND, k_bounce: float = K_BOUNCE, same_day_bounce: bool = False,
              point_in_time: bool = False) -> pd.DataFrame:
    """每个 as_of 日的支撑带、二段区间与 primed 标记（只使用 as_of 之前已收盘的日线）。

    point_in_time=False（主口径）：触及与反弹都按 as_of 当时的支撑带判定；
    point_in_time=True（S5）：日线 j 是否触及按 as_of = j 的支撑带判定，其后的反弹按触及日的支撑带判定。
    """
    day_ns = np.asarray(pd.DatetimeIndex(daily.index).as_unit("ns").asi8, dtype=np.int64)
    dlo, dhi = daily["low"].to_numpy(dtype=np.float64), daily["high"].to_numpy(dtype=np.float64)
    prof_ns = np.asarray(pd.DatetimeIndex(prof.index).as_unit("ns").asi8, dtype=np.int64)
    pos_prof = {int(x): i for i, x in enumerate(prof_ns)}
    pos_day = {int(x): i for i, x in enumerate(day_ns)}
    atr_v = atr.reindex(daily.index).to_numpy(dtype=np.float64)
    VAL, POC = prof["VAL"].to_numpy(), prof["POC"].to_numpy()
    out = np.full((len(prof), 6), np.nan)
    primed = np.zeros(len(prof), dtype=bool)
    for i, D in enumerate(prof_ns):
        j_prev = pos_day.get(int(D) - DAY_NS)          # as_of − 1d：最后一根已收盘日线
        if j_prev is None or not np.isfinite(VAL[i]) or not np.isfinite(atr_v[j_prev]):
            continue
        a = atr_v[j_prev]
        b_lo, b_hi = VAL[i] - k_band * a, VAL[i] + k_band * a
        out[i] = (a, b_lo, b_hi, b_hi + k_bounce * a, POC[i], VAL[i])
        j0 = j_prev - touch_days + 1
        if j0 < 0 or day_ns[j0] != int(D) - touch_days * DAY_NS:
            continue                                     # 触及回看窗口不足 30 根日线
        if not point_in_time:
            touch = (dlo[j0:j_prev + 1] <= b_hi) & (dhi[j0:j_prev + 1] >= b_lo)
            bounce = dhi[j0:j_prev + 1] >= b_hi + k_bounce * a
            if touch.any():
                f = int(np.argmax(touch))
                primed[i] = bool(bounce[f:].any()) if same_day_bounce else bool(bounce[f + 1:].any())
        else:
            for j1 in range(j0, j_prev + 1):
                k = pos_prof.get(int(day_ns[j1]))         # as_of = j1 的支撑带（用 j1 之前的数据）
                if k is None or not np.isfinite(out[k, 1]):
                    continue
                lo1, hi1, bnc1 = out[k, 1], out[k, 2], out[k, 3]
                if dlo[j1] <= hi1 and dhi[j1] >= lo1:
                    rest = dhi[j1:j_prev + 1] if same_day_bounce else dhi[j1 + 1:j_prev + 1]
                    if (rest >= bnc1).any():
                        primed[i] = True
                        break
    st = pd.DataFrame(out, columns=["ATR", "band_lo", "band_hi", "bounce_level", "POC", "VAL"], index=prof.index)
    st["primed"] = primed
    st["valid"] = np.isfinite(st["band_lo"].to_numpy()) & np.isfinite(st["POC"].to_numpy())
    return st


def minute_box_ok(minute_open_ns: np.ndarray, ref_price: np.ndarray, state: pd.DataFrame) -> np.ndarray:
    """box_ok(T) = primed(as_of(T)) ∧ band_lo ≤ ref ≤ POC，as_of(T) = floor_day(T)；无状态的分钟为 False。"""
    st_ns = np.asarray(pd.DatetimeIndex(state.index).as_unit("ns").asi8, dtype=np.int64)
    as_of = (minute_open_ns // DAY_NS) * DAY_NS
    k = np.searchsorted(st_ns, as_of)
    k_ok = (k < len(st_ns)) & (st_ns[np.minimum(k, len(st_ns) - 1)] == as_of)
    k = np.minimum(k, len(st_ns) - 1)
    primed = state["primed"].to_numpy()[k] & state["valid"].to_numpy()[k] & k_ok
    lo, poc = state["band_lo"].to_numpy()[k], state["POC"].to_numpy()[k]
    with np.errstate(invalid="ignore"):
        return primed & (ref_price >= lo) & (ref_price <= poc)


def build_box_states(df1m: pd.DataFrame, spot_1d: pd.DataFrame | None, spot_1m: pd.DataFrame | None = None,
                     variants: tuple[str, ...] = ("main",)) -> dict:
    """主口径与各敏感性的 as_of 日状态表。返回 {name: state DataFrame}，并附 'daily'（永续日线）。"""
    daily = daily_from_1m(df1m)
    atr = atr_1d(daily, spot_1d)
    first_perp = pd.DatetimeIndex(df1m.index).as_unit("ns")[0]
    # as_of 日 = 有 1m 数据的每个自然日（含数据末端尚未收盘的那一天：它的状态只用此前已收盘的日线）
    as_of = pd.DatetimeIndex(daily.index).append(pd.DatetimeIndex([daily.index[-1] + DAY])) \
        if len(daily) else pd.DatetimeIndex([first_perp.floor("D")])
    as_of = as_of[as_of <= pd.DatetimeIndex(df1m.index).as_unit("ns")[-1].floor("D")]
    out = {"daily": daily, "atr": atr}
    cache = {}

    def prof(source: str) -> pd.DataFrame:
        if source not in cache:
            if source == "1m":
                cache[source] = rolling_profiles(df1m, as_of, min_first_bar=first_perp)
            elif source == "1h":
                cache[source] = rolling_profiles(agg_ohlcv(df1m, "1h"), as_of, min_first_bar=first_perp)
            elif source == "1d":
                cache[source] = rolling_profiles(daily, as_of, min_first_bar=first_perp)
            elif source == "1m_spot_stitched":
                cols = ["open", "high", "low", "close", "volume"]
                x = pd.concat([spot_1m[spot_1m.index < first_perp][cols], df1m[cols]])
                cache[source] = rolling_profiles(x, as_of, min_first_bar=pd.DatetimeIndex(spot_1m.index)[0].ceil("D"))
        return cache[source]

    for v in variants:
        if v in ("main", "S3_origin_ref"):
            out[v] = box_state(daily, prof("1m"), atr)
        elif v == "S1a_profile_1h":
            out[v] = box_state(daily, prof("1h"), atr)
        elif v == "S1b_profile_1d":
            out[v] = box_state(daily, prof("1d"), atr)
        elif v == "S2_same_day_bounce":
            out[v] = box_state(daily, prof("1m"), atr, same_day_bounce=True)
        elif v == "S4_spot_stitched_warmup":
            out[v] = box_state(_stitched_daily(daily, spot_1d), prof("1m_spot_stitched"), _stitched_atr(daily, spot_1d))
        elif v == "S5_point_in_time":
            out[v] = box_state(daily, prof("1m"), atr, point_in_time=True)
        else:
            raise ValueError(v)
    return out


def _stitched_daily(daily: pd.DataFrame, spot_1d: pd.DataFrame) -> pd.DataFrame:
    w = spot_1d[spot_1d.index < daily.index[0]][["open", "high", "low", "close", "volume"]].copy()
    w["n_min"] = 1440
    w["available_time"] = w.index + DAY
    return pd.concat([w, daily])


def _stitched_atr(daily: pd.DataFrame, spot_1d: pd.DataFrame) -> pd.Series:
    sd = _stitched_daily(daily, spot_1d)
    a = wilder_atr(sd["high"].to_numpy(), sd["low"].to_numpy(), sd["close"].to_numpy(), ATR_LEN_1D)
    return pd.Series(a, index=sd.index)


def minutes_to_ns(idx) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(idx).as_unit("ns").asi8, dtype=np.int64)


assert MINUTE_NS == 60_000_000_000

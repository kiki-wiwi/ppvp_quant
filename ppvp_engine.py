"""
ppvp_engine.py — 4H 枢轴点成交量分布（Pivot Points Volume Profile, PPVP）引擎，零未来函数实现。

时间与可见性规则（全部在代码中强制执行，并由 test_lookahead.py 验证）
------------------------------------------------------------------
1. 4H K 线由 1H 按 UTC 绝对边界 00/04/08/12/16/20 聚合（``resample(origin='epoch')``），
   严禁按行号等距切片；含维护缺口的 4H 由剩余有效 1H 聚合（``n_bars`` 记录根数）。
   ``available_time = open_time + 4h``；数据末端尚未走完的 4H（available_time > 最后一根 1H 收盘）被剔除。
2. 位于 4H 索引 i 的枢轴点只用 [i-L, i+R] 的 K 线判定，且仅在第 i+R 根 4H 完整收盘后
   （``confirm_time = available_time[i+R]``）才被视为已知。
3. 箱体 = 最近 ``profile_pivots`` 个已确认枢轴之间的窗口 [open(4H_a), available(4H_b))，
   用窗口内已收盘的 1H K 线把成交量按 [low, high] 均匀分配到 ``n_bins`` 个价格桶，
   最大桶中心为 POC，自 POC 向两侧逐桶扩展至累计 ``va_pct`` 得到 VAL / VAH。
   箱体 ``available_time`` = 其最后一个枢轴的确认时间。
4. 1H K 线 i 只使用 ``available_time <= open_time_1h[i]`` 的箱体（即该 1H K 线开始前已经确认的箱体）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from config import StrategyConfig
from data_loader import resample_ohlcv
from numba_compat import njit

H4 = pd.Timedelta(hours=4)


# ----------------------------------------------------------------------------
# 时间工具
# ----------------------------------------------------------------------------
def to_ns(x) -> np.ndarray:
    """DatetimeIndex / Series → int64 纳秒（与 pandas 存储分辨率 ms/us/ns 无关）。"""
    if isinstance(x, pd.Series):
        x = pd.DatetimeIndex(x)
    return np.asarray(pd.DatetimeIndex(x).as_unit("ns").asi8, dtype=np.int64)


# ----------------------------------------------------------------------------
# 4H 聚合（UTC 绝对边界）
# ----------------------------------------------------------------------------
def build_4h(df1h: pd.DataFrame) -> pd.DataFrame:
    """1H → 4H，严格锚定 UTC 00/04/08/12/16/20 边界，只保留已完整收盘的 4H。

    返回列: open/high/low/close/volume/quote_volume/n_bars/close_time，以及
    ``available_time``（= open_time + 4h，即该 4H 对 1H 策略可见的最早时刻）。
    """
    cols = ["open", "high", "low", "close", "volume", "quote_volume"]
    df4 = resample_ohlcv(df1h[cols], "4h")
    df4["available_time"] = df4.index + H4
    last_1h_close = df1h.index[-1] + pd.Timedelta(hours=1)
    return df4.loc[df4["available_time"] <= last_1h_close].copy()


# ----------------------------------------------------------------------------
# Wilder ATR（因果递推）
# ----------------------------------------------------------------------------
@njit(cache=True)
def _wilder_atr(high, low, close, n):
    m = high.shape[0]
    out = np.full(m, np.nan)
    if m == 0:
        return out
    tr = np.empty(m)
    tr[0] = high[0] - low[0]
    for i in range(1, m):
        a = high[i] - low[i]
        b = abs(high[i] - close[i - 1])
        c = abs(low[i] - close[i - 1])
        tr[i] = max(a, max(b, c))
    if m < n:
        return out
    s = 0.0
    for i in range(n):
        s += tr[i]
    out[n - 1] = s / n
    for i in range(n, m):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """ATR[i] 只依赖 [0, i] 的 K 线（含第 i 根），在第 i 根收盘时可知。"""
    return _wilder_atr(np.ascontiguousarray(high, dtype=np.float64), np.ascontiguousarray(low, dtype=np.float64),
                       np.ascontiguousarray(close, dtype=np.float64), int(n))


def shift_known_at_open(x: np.ndarray) -> np.ndarray:
    """把“第 i 根收盘时可知”的序列右移一位，得到“第 i 根开盘时可知”的序列。"""
    out = np.empty_like(x, dtype=np.float64)
    out[0] = np.nan
    out[1:] = x[:-1]
    return out


# ----------------------------------------------------------------------------
# 枢轴点（右侧 R 根确认滞后）
# ----------------------------------------------------------------------------
def detect_pivots(high: np.ndarray, low: np.ndarray, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (is_pivot_high, is_pivot_low)。

    枢轴高点：high[i] 严格大于左侧 L 根的最高价，且 ≥ 右侧 R 根的最高价（平顶取第一根）。
    枢轴低点对称。索引 i 的判定只用到 [i-L, i+R]；末尾不足 R 根右侧 K 线的位置一律为 False，
    因此在第 i+R 根 4H 收盘前，i 处不可能被识别为枢轴。
    """
    n = len(high)
    is_ph = np.zeros(n, dtype=bool)
    is_pl = np.zeros(n, dtype=bool)
    w = left + right + 1
    if n < w:
        return is_ph, is_pl
    wh = sliding_window_view(high, w)
    wl = sliding_window_view(low, w)
    ch, cl = wh[:, left], wl[:, left]
    lmax = wh[:, :left].max(axis=1) if left > 0 else np.full(len(ch), -np.inf)
    rmax = wh[:, left + 1:].max(axis=1) if right > 0 else np.full(len(ch), -np.inf)
    lmin = wl[:, :left].min(axis=1) if left > 0 else np.full(len(cl), np.inf)
    rmin = wl[:, left + 1:].min(axis=1) if right > 0 else np.full(len(cl), np.inf)
    is_ph[left:n - right] = (ch > lmax) & (ch >= rmax)
    is_pl[left:n - right] = (cl < lmin) & (cl <= rmin)
    return is_ph, is_pl


# ----------------------------------------------------------------------------
# 成交量分布（1H → 价格分桶）
# ----------------------------------------------------------------------------
@njit(cache=True)
def _value_area(hist, va_pct):
    """自 POC 桶起逐桶向量较大的一侧扩展（平局向上），直到累计量 ≥ va_pct * 总量。返回 (poc, lo, hi) 桶号。"""
    nb = hist.shape[0]
    poc = 0
    for j in range(1, nb):
        if hist[j] > hist[poc]:
            poc = j
    total = 0.0
    for j in range(nb):
        total += hist[j]
    target = va_pct * total
    lo = poc
    hi = poc
    acc = hist[poc]
    while acc < target:
        up = hist[hi + 1] if hi + 1 < nb else -1.0
        dn = hist[lo - 1] if lo - 1 >= 0 else -1.0
        if up < 0.0 and dn < 0.0:
            break
        if up >= dn:
            hi += 1
            acc += up
        else:
            lo -= 1
            acc += dn
    return poc, lo, hi


def volume_profile(lo: np.ndarray, hi: np.ndarray, vol: np.ndarray, n_bins: int, va_pct: float) -> dict:
    """把每根 1H 的成交量按其 [low, high] 区间均匀分配到价格桶，返回 POC/VAL/VAH/Profile_Low/High 与直方图。"""
    p_low, p_high = float(lo.min()), float(hi.max())
    if not p_high > p_low:
        return {"poc": p_low, "val": p_low, "vah": p_high, "p_low": p_low, "p_high": p_high,
                "hist": np.array([vol.sum()]), "edges": np.array([p_low, p_high])}
    edges = np.linspace(p_low, p_high, n_bins + 1)
    rng = hi - lo
    pos = rng > 0
    ov = np.minimum(hi[:, None], edges[None, 1:]) - np.maximum(lo[:, None], edges[None, :-1])
    ov = np.clip(ov, 0.0, None)
    w = np.zeros_like(ov)
    w[pos] = ov[pos] / rng[pos, None]
    hist = (w * vol[:, None]).sum(axis=0)
    if (~pos).any():                       # 一字线：全部成交量计入价格所在桶
        width = (p_high - p_low) / n_bins
        idx = np.clip(((lo[~pos] - p_low) / width).astype(np.int64), 0, n_bins - 1)
        np.add.at(hist, idx, vol[~pos])
    poc_i, lo_i, hi_i = _value_area(hist, float(va_pct))
    return {"poc": 0.5 * (edges[poc_i] + edges[poc_i + 1]), "val": float(edges[lo_i]), "vah": float(edges[hi_i + 1]),
            "p_low": p_low, "p_high": p_high, "hist": hist, "edges": edges}


# ----------------------------------------------------------------------------
# 箱体表
# ----------------------------------------------------------------------------
def build_boxes(df1h: pd.DataFrame, cfg: StrategyConfig, df4h: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构建全部 PPVP 箱体（每个新确认枢轴生成一个箱体），返回 (boxes, df4h)。

    boxes 每行: available_time（箱体可被 1H 使用的最早时刻）、两端枢轴（4H 索引/时间/类型）、
    窗口 [window_start, window_end)、1H 根数、VAL/POC/VAH/Profile_Low/Profile_High、总成交量。
    """
    if df4h is None:
        df4h = build_4h(df1h)
    L, R, P = cfg.pivot_left, cfg.pivot_right, cfg.profile_pivots
    h4, l4 = df4h["high"].to_numpy(), df4h["low"].to_numpy()
    open4 = to_ns(df4h.index)
    avail4 = to_ns(df4h["available_time"])
    is_ph, is_pl = detect_pivots(h4, l4, L, R)
    piv = np.flatnonzero(is_ph | is_pl)

    open1 = to_ns(df1h.index)
    lo1, hi1, v1 = (df1h[c].to_numpy(dtype=np.float64) for c in ("low", "high", "volume"))

    rows = []
    for k in range(P - 1, len(piv)):
        a, b = int(piv[k - P + 1]), int(piv[k])
        t0, t1 = open4[a], avail4[b]                      # 窗口 [4H_a 开盘, 4H_b 收盘)
        s = int(np.searchsorted(open1, t0, side="left"))
        e = int(np.searchsorted(open1, t1, side="left"))  # 只含 open_time < 4H_b 收盘 的 1H（均已收盘）
        if e - s < 1:
            continue
        prof = volume_profile(lo1[s:e], hi1[s:e], v1[s:e], cfg.n_bins, cfg.va_pct)
        typ = lambda j: ("HL" if (is_ph[j] and is_pl[j]) else "H" if is_ph[j] else "L")
        rows.append({
            "available_time": avail4[b + R],              # 第 b+R 根 4H 收盘 = 枢轴 b 确认时刻
            "pivot_a_idx": a, "pivot_b_idx": b, "pivot_a_type": typ(a), "pivot_b_type": typ(b),
            "pivot_a_time": open4[a], "pivot_b_time": open4[b],
            "window_start": t0, "window_end": t1, "n_1h": e - s,
            "VAL": prof["val"], "POC": prof["poc"], "VAH": prof["vah"],
            "P_low": prof["p_low"], "P_high": prof["p_high"], "total_volume": float(v1[s:e].sum()),
        })
    boxes = pd.DataFrame(rows)
    if len(boxes):
        for c in ("available_time", "pivot_a_time", "pivot_b_time", "window_start", "window_end"):
            boxes[c] = pd.to_datetime(boxes[c].to_numpy(dtype=np.int64), unit="ns", utc=True)
        boxes.index.name = "box_id"
    return boxes, df4h


def map_boxes_to_1h(df1h: pd.DataFrame, boxes: pd.DataFrame) -> np.ndarray:
    """返回每根 1H 可用的最新箱体编号（-1 = 尚无箱体）：available_time ≤ 该 1H 的 open_time。"""
    if len(boxes) == 0:
        return np.full(len(df1h), -1, dtype=np.int64)
    avail = to_ns(boxes["available_time"])
    assert np.all(np.diff(avail) > 0), "box available times must be strictly increasing"
    return (np.searchsorted(avail, to_ns(df1h.index), side="right") - 1).astype(np.int64)


def pivot_table(df4h: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """审计用：列出每个枢轴的 [发生时间, 确认可用时间]。"""
    is_ph, is_pl = detect_pivots(df4h["high"].to_numpy(), df4h["low"].to_numpy(), cfg.pivot_left, cfg.pivot_right)
    idx = np.flatnonzero(is_ph | is_pl)
    avail = df4h["available_time"].to_numpy()
    return pd.DataFrame({
        "pivot_bar_open": df4h.index[idx],
        "type": ["HL" if (is_ph[i] and is_pl[i]) else "H" if is_ph[i] else "L" for i in idx],
        "price": [df4h["high"].iloc[i] if is_ph[i] else df4h["low"].iloc[i] for i in idx],
        "confirm_time": avail[idx + cfg.pivot_right],
    })

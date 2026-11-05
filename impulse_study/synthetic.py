"""
impulse_study/synthetic.py — Checkpoint 2 的合成数据（单元测试与耗时基准；真实行情在本阶段不得进入撮合）。

* ``grid_from_arrays``   由手写的 O/H/L/C/V 构造分钟网格（与 data_1m.minute_grid 同列），bad 分钟按维护占位处理；
* ``random_grid``        带波动聚集、随机脉冲、成交量放大与维护停机的随机游走（耗时基准与随机化测试）；
* ``synthetic_context``  features.make_context 的合成数据入口（source = 'synthetic'）；
* ``manual_inputs``      直接给定逐分钟入场方向 / 止损 / E_ref 的版本输入（只测撮合与记账时使用）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import versions as VS
from .features import Context, VersionInputs, make_context


def grid_from_arrays(start, O, H, L, C, V, bad=None) -> pd.DataFrame:
    n = len(C)
    idx = pd.date_range(pd.Timestamp(start), periods=n, freq="1min").as_unit("ns")
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    bad = np.zeros(n, dtype=bool) if bad is None else np.asarray(bad, dtype=bool)
    g = pd.DataFrame({"open": np.asarray(O, float), "high": np.asarray(H, float), "low": np.asarray(L, float),
                      "close": np.asarray(C, float), "volume": np.asarray(V, float)}, index=idx)
    g.loc[bad, "volume"] = 0.0
    g["quote_volume"] = g["volume"] * g["close"]
    g["count"] = np.where(g["volume"] > 0, 10.0, 0.0)
    g["taker_buy_volume"] = g["volume"] / 2
    g.index.name = "open_time"
    g["missing"] = False
    g["zero_volume"] = (g["volume"] == 0).to_numpy()
    g["placeholder"] = bad
    g["bad"] = bad
    return g


def flat_bars(n: int, price: float = 10_000.0, vol: float = 10.0, spread: float = 0.0):
    """价格恒定的基础序列（手算测试在其上改写个别分钟）。"""
    O = np.full(n, price)
    C = np.full(n, price)
    H = np.full(n, price * (1 + spread))
    L = np.full(n, price * (1 - spread))
    V = np.full(n, vol)
    return O, H, L, C, V


def random_grid(n: int, start: str = "2020-01-01", seed: int = 0, price0: float = 10_000.0, sigma: float = 6e-4,
                impulse_rate: float = 4e-4, halts: tuple = ()) -> pd.DataFrame:
    """随机游走：对数波动率 AR(1) 聚集；以 impulse_rate 的概率在某分钟起出现持续 1–10 分钟、总幅 0.4%–2.5% 的单向脉冲。"""
    rng = np.random.default_rng(seed)
    lv = np.empty(n)
    lv[0] = 0.0
    eps = rng.normal(0, 0.05, n)
    for t in range(1, n):
        lv[t] = 0.999 * lv[t - 1] + eps[t]
    sig = sigma * np.exp(np.clip(lv, -1.5, 1.5))
    r = rng.standard_t(5, n) * sig / np.sqrt(5 / 3)
    vol_mult = np.ones(n)
    starts = np.flatnonzero(rng.random(n) < impulse_rate)
    for s0 in starts:
        dur = int(rng.integers(1, 11))
        tot = rng.uniform(0.004, 0.025) * rng.choice([-1.0, 1.0])
        r[s0:s0 + dur] += tot / dur
        vol_mult[s0:s0 + dur + 15] *= rng.uniform(2.0, 6.0)
    C = price0 * np.exp(np.cumsum(r))
    O = np.empty(n)
    O[0] = price0
    O[1:] = C[:-1] * np.exp(rng.normal(0, sig[1:] * 0.05))
    wick = np.abs(rng.normal(0, sig, (2, n)))
    H = np.maximum(O, C) * np.exp(wick[0])
    L = np.minimum(O, C) * np.exp(-wick[1])
    V = rng.lognormal(3.0, 0.6, n) * vol_mult
    bad = np.zeros(n, dtype=bool)
    for h0, hl in halts:
        bad[h0:h0 + hl] = True
        px = C[h0 - 1]
        O[h0:h0 + hl] = H[h0:h0 + hl] = L[h0:h0 + hl] = C[h0:h0 + hl] = px
    return grid_from_arrays(start, O, H, L, C, V, bad)


def synthetic_context(grid: pd.DataFrame, main_start, last_trigger, box_ok=None, funding: pd.Series | None = None,
                      archive_gap_minutes=()) -> Context:
    box = np.zeros(len(grid), dtype=bool) if box_ok is None else np.asarray(box_ok, dtype=bool)
    return make_context(grid, box, "synthetic", main_start=main_start, last_trigger=last_trigger,
                        archive_gap_minutes=archive_gap_minutes, funding=funding)


def manual_inputs(ctx: Context, spec: VS.VersionSpec, signals: dict) -> VersionInputs:
    """signals：{触发分钟索引: (方向 ±1, 止损价)}；E_ref 取该分钟收盘价。只用于撮合与记账测试。"""
    n = len(ctx.C)
    d = np.zeros(n, dtype=np.int8)
    sl = np.full(n, np.nan)
    for i, (s, stop) in signals.items():
        d[i] = s
        sl[i] = stop
    return VersionInputs(spec, d, sl, ctx.C.copy(), np.zeros(n, np.int8), np.where(d != 0, 1, 0).astype(np.int8),
                         ctx.atr["atr_at"], {})

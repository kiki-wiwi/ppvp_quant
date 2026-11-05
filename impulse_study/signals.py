"""
impulse_study/signals.py — 时间价格脉冲触发器与纯频次统计（Checkpoint 1：不读取任何触发后的价格，不计算任何盈亏）。

触发器（全阶段统一标准；无量纲阈值 = 交易员在 85,000 USDT 附近监控的 150 / 400 / 1000 USDT，按规格取整）
--------------------------------------------------------------------------------------------------
每根 1m K 线 T 收盘（T + 1min）时，C_T 为其收盘价，r_k = C_T / C_{T−k} − 1（k = 1, 5, 15，按 UTC 分钟网格取 T−k 那一根的收盘价）：
  指标 ①  r1 ≥ +0.18% 且 r5 ≥ +0.47%（向下：r1 ≤ −0.18% 且 r5 ≤ −0.47%）
  指标 ②  r15 ≥ +1.18%（向下：r15 ≤ −1.18%）
  Up_Impulse = ① ∨ ②（向上），Down_Impulse 同理；拆分为“仅 ①”“仅 ②”“① + ② 同时”。
端点为缺失或维护占位分钟（零成交且 O = H = L = C）时收益率记为缺失、不触发；15 分钟回看 [T−15, T] 内含占位分钟的触发
记为“维护污染”，单独计数，不进入可交易计数。

稳健性口径（Z-score，另行登记）：σ7d(T) = 截至 T−15 的 10,080 根 1m 简单收益率的均方根（不含脉冲窗口本身，
有效值 ≥ 95%），三个阈值按 σ7d(T) / σ_ref 等比缩放；σ_ref = IS 全部分钟 σ7d 的中位数（只用价格波动，不用任何结果）。
在 σ7d = σ_ref 时两种口径相同。

计数层级
--------
raw     满足条件的分钟数（大行情中条件会连续多分钟成立）；
onset   条件由不成立变为成立的分钟数（独立“事件”数的上界近似）；
sched   按持仓互斥 + 冷却推算的可成交笔数：只用触发时间戳与固定的持仓日程，不读取触发后的价格。
        触发 K 线 T → T+1min 开盘入场；持 1 根合成 15m 棒于 T+16min 离场、冷却 15 分钟 → 下一触发须 ≥ T+30min（sched30）；
        持满 2 根于 T+31min 离场 → 下一触发须 ≥ T+45min（sched45）。实际笔数在两者之间（止损提前离场可多于 sched30）。
可交易窗口：触发 K 线 T ∈ [IS 起点 + 15min, IS_END − 31min]，保证 15 分钟回看与最长 30 分钟持仓都在 IS 之内。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scope import IS_END, IS_START, MINUTE_NS

THR = {"r1": 0.0018, "r5": 0.0047, "r15": 0.0118}
SIGMA_WINDOW = 10_080           # 7 日 1m 根数
SIGMA_MIN_VALID = 0.95
LOOKBACK = 15
MAX_HOLD_MIN = 30
BLOCK = {"sched30": 30, "sched45": 45}


def impulse_frame(grid: pd.DataFrame) -> pd.DataFrame:
    """grid = data_1m.minute_grid(...)（完整分钟网格 + 标记）。返回逐分钟的收益率、触发条件与污染标记。"""
    bad = grid["bad"].to_numpy()
    c = grid["close"].where(~grid["bad"])
    f = pd.DataFrame(index=grid.index)
    f["close"] = grid["close"]
    for k in (1, 5, 15):
        f[f"r{k}"] = c / c.shift(k) - 1.0
    # [T−15, T] 共 16 根内是否有缺失 / 占位分钟
    f["contaminated"] = pd.Series(bad.astype(np.int8), index=grid.index).rolling(LOOKBACK + 1, min_periods=1).max() \
        .to_numpy().astype(bool)
    r1sq = f["r1"] ** 2
    m = r1sq.rolling(SIGMA_WINDOW, min_periods=int(SIGMA_MIN_VALID * SIGMA_WINDOW)).mean()
    f["sigma7d"] = np.sqrt(m.shift(LOOKBACK))                 # 截至 T−15，不含脉冲窗口
    f["sigma60"] = np.sqrt(r1sq.rolling(60, min_periods=55).mean())   # 截至 T（含脉冲本身），只用于功效估计
    return f


def conditions(f: pd.DataFrame, scale: np.ndarray | float = 1.0) -> dict:
    """返回 {'up1','up2','dn1','dn2'} 布尔数组；scale 为阈值缩放（Z-score 口径 = σ7d / σ_ref，缺失时不触发）。"""
    r1, r5, r15 = (f[c].to_numpy() for c in ("r1", "r5", "r15"))
    s = np.broadcast_to(np.asarray(scale, dtype=np.float64), r1.shape)
    t1, t5, t15 = THR["r1"] * s, THR["r5"] * s, THR["r15"] * s
    with np.errstate(invalid="ignore"):
        return {"up1": (r1 >= t1) & (r5 >= t5), "up2": r15 >= t15,
                "dn1": (r1 <= -t1) & (r5 <= -t5), "dn2": r15 <= -t15}


def eligible_window(index: pd.DatetimeIndex) -> np.ndarray:
    t = np.asarray(pd.DatetimeIndex(index).as_unit("ns").asi8, dtype=np.int64)
    lo = int(IS_START.value) + LOOKBACK * MINUTE_NS
    hi = int(IS_END.value) - (MAX_HOLD_MIN + 1) * MINUTE_NS
    return (t >= lo) & (t <= hi)


def onset(mask: np.ndarray) -> np.ndarray:
    prev = np.concatenate([[False], mask[:-1]])
    return mask & ~prev


def schedule(mask: np.ndarray, block: int) -> np.ndarray:
    """贪心：接受第一个触发，其后 block 分钟内的触发全部忽略（分钟网格连续，索引差 = 分钟差）。"""
    idx = np.flatnonzero(mask)
    acc = np.zeros(mask.shape[0], dtype=bool)
    nxt = -1
    for i in idx:
        if i >= nxt:
            acc[i] = True
            nxt = i + block
    return acc


def split_counts(c1: np.ndarray, c2: np.ndarray, sel: np.ndarray) -> dict:
    return {"only_1": int((sel & c1 & ~c2).sum()), "only_2": int((sel & ~c1 & c2).sum()),
            "both": int((sel & c1 & c2).sum()), "total": int((sel & (c1 | c2)).sum())}


def years_of(index: pd.DatetimeIndex) -> np.ndarray:
    return pd.DatetimeIndex(index).year.to_numpy()


def direction_table(c1: np.ndarray, c2: np.ndarray, base: np.ndarray, box: np.ndarray, years: np.ndarray,
                    year_list=(2020, 2021, 2022, 2023)) -> dict:
    """单一方向：raw / onset / sched（无箱体与有箱体），按自然年与 IS 合计。base = 可交易窗口且未污染。"""
    trig = (c1 | c2) & base
    trig_box = trig & box
    on = onset(c1 | c2) & base
    on_box = on & box
    sets = {"sched30": schedule(trig, 30), "sched45": schedule(trig, 45),
            "box_sched30": schedule(trig_box, 30), "box_sched45": schedule(trig_box, 45)}
    out = {}
    for y in list(year_list) + ["IS"]:
        ys = np.ones_like(trig) if y == "IS" else (years == y)
        out[str(y)] = {"raw": split_counts(c1, c2, trig & ys), "onset": split_counts(c1, c2, on & ys),
                       "box_raw": split_counts(c1, c2, trig_box & ys), "box_onset": split_counts(c1, c2, on_box & ys),
                       **{k: split_counts(c1, c2, v & ys) for k, v in sets.items()}}
    return out, sets


def concentration(acc: np.ndarray, index: pd.DatetimeIndex) -> dict:
    t = pd.DatetimeIndex(index)[acc]
    n = int(acc.sum())
    if n == 0:
        return {"n": 0}
    ym = pd.Series(t.year * 100 + t.month).value_counts()
    dd = pd.Series(t.normalize()).value_counts()
    sh = (ym / n).to_numpy()
    top5 = ym.head(5)
    return {"n": n, "months_with_events": int(len(ym)), "days_with_events": int(len(dd)),
            "top5_months": {f"{k // 100}-{k % 100:02d}": int(v) for k, v in top5.items()},
            "top5_months_share": float(top5.sum() / n), "max_events_one_day": int(dd.max()),
            "max_day": str(dd.idxmax().date()), "effective_number_of_months": float(1.0 / (sh ** 2).sum())}


def mde_bps(sigma_1m: np.ndarray, horizon_min: int, n: int | None = None, alpha_z: float = 1.959964,
            power_z: float = 0.841621) -> float:
    """独立同分布近似下的最小可检测平均收益（双侧 5%、功效 80%）：(z_a + z_b) · RMS(σ_1m·√h) / √n，单位 bp。

    σ_1m 只取触发时刻之前（含触发 K 线）的波动率，不涉及任何触发后的价格。
    """
    s = sigma_1m[np.isfinite(sigma_1m)]
    n = len(s) if n is None else n
    if n == 0:
        return float("nan")
    rms_h = float(np.sqrt(np.mean(s ** 2)) * np.sqrt(horizon_min))
    return (alpha_z + power_z) * rms_h / np.sqrt(n) * 1e4

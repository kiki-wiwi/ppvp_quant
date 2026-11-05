"""
impulse_study/qa.py — Checkpoint 1：1m 数据完整性与跨周期交叉对齐核验（只读数据，不计算任何策略结果）。

* ``qa_1m_integrity``     分钟网格完整性、重复 / 乱序 / 越格、OHLC 逻辑、收盘时间、零成交与维护占位分钟、极端收益；
* ``qa_vs_reference``     1m 按 UTC 边界聚合后与参照 K 线（仓库 1H、官方 15m、官方 1d）逐根比对 OHLCV；
* ``qa_daily_available``  UTC 日线 ``available_time = open_time + 24h`` 与官方日线收盘时间的逐根核验；
* ``qa_spot_warmup``      现货预热数据的完整性及与仓库现货 1H 的一致性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_1m import INTERVAL_MS, PRICE_COLS, agg_ohlcv, minute_grid
from .scope import DAY, IS_END, IS_START, MINUTE


def _ranges(ts: pd.DatetimeIndex, step: pd.Timedelta) -> list[list]:
    """离散时间戳 → 连续区间 [[first, last, count], ...]。"""
    if len(ts) == 0:
        return []
    out, s, prev, n = [], ts[0], ts[0], 1
    for t in ts[1:]:
        if t - prev == step:
            prev, n = t, n + 1
        else:
            out.append([str(s), str(prev), n])
            s, prev, n = t, t, 1
    out.append([str(s), str(prev), n])
    return out


def _epoch_ms(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(idx).as_unit("ns").asi8 // 1_000_000, dtype=np.int64)


def qa_1m_integrity(df: pd.DataFrame, start: pd.Timestamp = IS_START, end: pd.Timestamp = IS_END) -> dict:
    idx = df.index
    grid = minute_grid(df, start, end)
    missing = grid.index[grid["missing"].to_numpy()]
    zv = grid.index[grid["zero_volume"].to_numpy()]
    ph = grid.index[grid["placeholder"].to_numpy()]
    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)
    bad_ohlc = (df["high"] < body_hi) | (df["low"] > body_lo) | (df["high"] < df["low"])
    tb = df["taker_buy_volume"] > df["volume"] * (1 + 1e-12)
    pos = df["volume"] > 0
    vwap = df["quote_volume"].where(pos) / df["volume"].where(pos)
    vwap_out = pos & ((vwap < df["low"] * (1 - 1e-9)) | (vwap > df["high"] * (1 + 1e-9)))
    lr = np.log(grid["close"]).diff()
    lr_abs = lr.abs()
    ym = lambda ix: pd.Series(pd.DatetimeIndex(ix).year * 100 + pd.DatetimeIndex(ix).month)
    by_month = ym(idx).value_counts()
    exp_month = ym(grid.index).value_counts()
    month_short = {f"{m // 100}-{m % 100:02d}": int(exp_month[m] - by_month.get(m, 0))
                   for m in sorted(exp_month.index) if exp_month[m] != by_month.get(m, 0)}
    res = {
        "rows": int(len(df)), "expected_rows": int(len(grid)), "first_open_time": str(idx.min()),
        "last_open_time": str(idx.max()), "duplicates": int(idx.duplicated().sum()),
        "monotonic_increasing": bool(idx.is_monotonic_increasing),
        "off_grid_rows": int((pd.DatetimeIndex(idx).as_unit("ns").asi8 % 60_000_000_000 != 0).sum()),
        "missing_minutes": int(len(missing)), "missing_ranges": _ranges(missing, MINUTE),
        "months_with_row_shortfall": month_short,
        "nan_cells": int(df[["open", "high", "low", "close", "volume"]].isna().sum().sum()),
        "bad_ohlc_rows": int(bad_ohlc.sum()), "non_positive_price_rows": int((df[PRICE_COLS] <= 0).any(axis=1).sum()),
        "negative_volume_rows": int((df["volume"] < 0).sum()),
        "zero_volume_minutes": int(len(zv)), "zero_volume_ranges": _ranges(zv, MINUTE),
        "placeholder_minutes": int(len(ph)), "placeholder_ranges": _ranges(ph, MINUTE),
        "zero_trade_count_rows": int((df["count"] == 0).sum()),
        "taker_buy_gt_volume_rows": int(tb.sum()), "vwap_outside_high_low_rows": int(vwap_out.sum()),
        "max_abs_1m_log_return": float(lr_abs.max()), "max_abs_1m_log_return_at": str(lr_abs.idxmax()),
        "minutes_abs_1m_return_gt_2pct": int((lr_abs > np.log(1.02)).sum()),
        "rows_by_src": {str(k): int(v) for k, v in df["src"].value_counts().sort_index().items()} if "src" in df else {},
    }
    if "raw_close_time_ms" in df.columns:
        res["close_time_mismatch_rows"] = int((df["raw_close_time_ms"].to_numpy() != _epoch_ms(idx) + 59_999).sum())
    return res


def qa_vs_reference(df1m: pd.DataFrame, ref: pd.DataFrame, rule: str, name: str) -> dict:
    """1m 按 UTC 边界（origin=epoch, label=left, closed=left）聚合到 rule 后，与参照 K 线逐根比对。

    价格要求逐位相等；成交量类字段报告最大相对差与超过 1e-9 的根数，以及“四舍五入到 0.001 BTC 后不相等”的根数。
    """
    agg = agg_ohlcv(df1m, rule)
    common = agg.index.intersection(ref.index)
    a, b = agg.loc[common], ref.loc[common]
    out = {"reference": name, "rule": rule, "bars_reference": int(len(ref)), "bars_aggregated": int(len(agg)),
           "bars_common": int(len(common)), "only_in_aggregated": [str(t) for t in agg.index.difference(ref.index)][:20],
           "only_in_reference": [str(t) for t in ref.index.difference(agg.index)][:20]}
    step_min = INTERVAL_MS[rule.lower()] // 60_000
    out["aggregated_bars_with_lt_full_minutes"] = int((a["n_min"] < step_min).sum())
    for c in PRICE_COLS:
        out[f"{c}_mismatch_bars"] = int((a[c].to_numpy() != b[c].to_numpy()).sum())
    bad = common[(a[PRICE_COLS].to_numpy() != b[PRICE_COLS].to_numpy()).any(axis=1)]
    out["ohlc_mismatch_bars_list"] = [str(t) for t in bad[:20]]
    for c in ("volume", "quote_volume", "taker_buy_volume", "count"):
        if c not in ref.columns:
            continue
        den = b[c].where(b[c] > 0)
        rel = ((a[c] - b[c]).abs() / den)
        out[f"{c}_max_rel_diff"] = float(rel.max()) if rel.notna().any() else 0.0
        out[f"{c}_bars_rel_gt_1e-9"] = int((rel > 1e-9).sum())
        if c in ("volume", "taker_buy_volume"):
            out[f"{c}_bars_unequal_after_round_0.001"] = int((a[c].round(3) != b[c].round(3)).sum())
        if c == "count":
            out["count_unequal_bars"] = int((a[c] != b[c]).sum())
    vbad = common[((a["volume"] - b["volume"]).abs() / b["volume"].where(b["volume"] > 0) > 1e-9).to_numpy()]
    out["volume_mismatch_bars_list"] = [[str(t), float(a.loc[t, "volume"]), float(b.loc[t, "volume"])] for t in vbad[:20]]
    if "raw_close_time_ms" in ref.columns:
        out["reference_close_time_mismatch_rows"] = int(
            (ref["raw_close_time_ms"].to_numpy() != _epoch_ms(ref.index) + INTERVAL_MS[rule.lower()] - 1).sum())
    return out


def daily_bars(df1m: pd.DataFrame) -> pd.DataFrame:
    """1m → UTC 日线（00:00 开盘），附 available_time = open_time + 24h（该日线完整收盘、可被使用的最早时刻）。"""
    d = agg_ohlcv(df1m, "1D")
    d["available_time"] = d.index + DAY
    return d


def qa_daily_available(d1: pd.DataFrame, official_1d: pd.DataFrame) -> dict:
    idx = pd.DatetimeIndex(d1.index).as_unit("ns")
    at = pd.DatetimeIndex(d1["available_time"]).as_unit("ns")
    out = {"days": int(len(d1)), "first_day": str(idx.min()), "last_day": str(idx.max()),
           "open_time_not_midnight_utc": int((idx.asi8 % (86_400 * 10**9) != 0).sum()),
           "available_time_ne_open_plus_24h": int((at.asi8 - idx.asi8 != 86_400 * 10**9).sum()),
           "days_with_lt_1440_minutes": int((d1["n_min"] < 1440).sum()),
           "consecutive_days_gap": int((np.diff(idx.asi8) != 86_400 * 10**9).sum()),
           "last_available_time": str(at.max()),
           "last_available_time_le_is_end": bool(at.max() <= IS_END)}
    if "raw_close_time_ms" in official_1d.columns:
        oc = official_1d["raw_close_time_ms"].to_numpy()
        out["official_1d_close_time_eq_open_plus_24h_minus_1ms"] = int((oc == _epoch_ms(official_1d.index) + 86_399_999).sum())
        out["official_1d_rows"] = int(len(official_1d))
        out["official_1d_open_time_not_midnight_utc"] = int(
            (pd.DatetimeIndex(official_1d.index).as_unit("ns").asi8 % (86_400 * 10**9) != 0).sum())
    return out


def qa_spot_warmup(s1d: pd.DataFrame, s1m: pd.DataFrame, spot_1h_repo: pd.DataFrame) -> dict:
    """现货预热：日线连续性、1m 缺失分钟，以及与仓库现货 1H（同一官方来源、Phase 0 已校验）聚合日线的一致性。"""
    ref = agg_ohlcv(spot_1h_repo.assign(count=spot_1h_repo.get("count", 0.0),
                                        taker_buy_volume=spot_1h_repo.get("taker_buy_volume", 0.0)), "1D")
    common = s1d.index.intersection(ref.index)
    a, b = s1d.loc[common], ref.loc[common]
    g = minute_grid(s1m, s1m.index.min().floor("D"), s1m.index.max().floor("D") + DAY)
    miss = g.index[g["missing"].to_numpy()]
    return {"spot_1d_rows": int(len(s1d)), "spot_1d_first": str(s1d.index.min()), "spot_1d_last": str(s1d.index.max()),
            "spot_1d_gaps": int((np.diff(pd.DatetimeIndex(s1d.index).as_unit("ns").asi8) != 86_400 * 10**9).sum()),
            "spot_1d_vs_repo_spot_1h_days_compared": int(len(common)),
            "spot_1d_vs_repo_spot_1h_ohlc_mismatch_days": [str(t) for t in common[
                (a[PRICE_COLS].to_numpy() != b[PRICE_COLS].to_numpy()).any(axis=1)]],
            "spot_1m_rows": int(len(s1m)), "spot_1m_missing_minutes": int(len(miss)),
            "spot_1m_missing_ranges": _ranges(miss, MINUTE),
            "spot_1m_zero_volume_minutes": int(g["zero_volume"].sum())}


def localize_volume_discrepancies(df1m: pd.DataFrame, ref_1h: pd.DataFrame, tol: float = 1e-9) -> dict:
    """1m 聚合与参照 1H 成交量不一致的小时：取该日官方 aggTrades 逐分钟定位差额，并比较该分钟 K 线与逐笔的高低收。

    只读数据，不改动 1m 缓存（官方 1m 产品原样保留，差异在此登记，供后续阶段对受影响分钟做标注）。
    """
    from .data_1m import fetch_aggtrades_1m

    agg_h = agg_ohlcv(df1m, "1h")
    common = agg_h.index.intersection(ref_1h.index)
    rel = (agg_h.loc[common, "volume"] - ref_1h.loc[common, "volume"]).abs() / ref_1h.loc[common, "volume"]
    hours = common[(rel > tol).to_numpy()]
    out = {"hours": [], "minutes": []}
    for day in sorted({t.floor("D") for t in hours}):
        agg, src = fetch_aggtrades_1m(day)
        kl = df1m[(df1m.index >= day) & (df1m.index < day + DAY)]
        dv = agg["volume"].reindex(kl.index).fillna(0.0) - kl["volume"]
        for h in [t for t in hours if t.floor("D") == day]:
            hk = kl[(kl.index >= h) & (kl.index < h + pd.Timedelta(hours=1))]
            ha = agg[(agg.index >= h) & (agg.index < h + pd.Timedelta(hours=1))]
            out["hours"].append({"hour": str(h), "volume_1m": float(hk["volume"].sum()),
                                 "volume_ref_1h": float(ref_1h.loc[h, "volume"]),
                                 "volume_aggtrades": float(ha["volume"].sum()), "aggtrades_url": src["url"]})
            dh = dv[(dv.index >= h) & (dv.index < h + pd.Timedelta(hours=1))]
            for t in dh.index[(dh.abs() > 1e-6).to_numpy()]:
                k, a = kl.loc[t], agg.loc[t] if t in agg.index else None
                hlc_diff = a is None or bool(a["high"] != k["high"] or a["low"] != k["low"] or a["close"] != k["close"])
                if abs(float(dh[t])) <= 0.5 and not hlc_diff:
                    continue            # 背景差异：少量成交被归入相邻分钟（成对出现、相互抵消），价格不变
                out["minutes"].append({
                    "minute": str(t), "volume_aggtrades_minus_1m": float(dh[t]),
                    "kline_hlc": [float(k["high"]), float(k["low"]), float(k["close"])],
                    "aggtrades_hlc": None if a is None else [float(a["high"]), float(a["low"]), float(a["close"])],
                    "extreme_differs": None if a is None else bool(a["high"] != k["high"] or a["low"] != k["low"]),
                    "close_differs": None if a is None else bool(a["close"] != k["close"])})
    return out

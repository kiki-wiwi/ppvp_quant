"""
impulse_study/qa.py — Checkpoint 1：1m 数据完整性与跨周期交叉对齐核验（只读数据，不计算任何策略结果）。

* ``qa_1m_integrity``                 分钟网格完整性、重复 / 乱序 / 越格、OHLC 逻辑、收盘时间、零成交与维护占位分钟、极端收益；
* ``qa_vs_reference``                 1m 按 UTC 边界聚合后与参照 K 线（仓库 1H、官方 15m、官方 1d）逐根比对 OHLCV；
                                      成交量按绝对差判定，参照为 0 而 1m 有成交的 K 线单独计数（不会被相对差的除零漏掉）；
* ``qa_daily_available``              UTC 日线的可用时间：由 1m 数据独立核对“每根日线的最后一分钟在 open_time + 24h 收盘”，
                                      以及官方日线收盘时间；
* ``qa_spot_warmup``                  现货预热数据的完整性及与仓库现货 1H 的一致性；
* ``localize_volume_discrepancies``   1m 与 1H（逐小时）或官方 1d（逐日）成交量不一致的自然日：用官方逐笔归档逐分钟定位，
                                      对价格与逐笔不同的分钟做“分钟末尾成交被截断”的前缀分析；
* ``monthly_vs_daily_check``          指定自然日的月度 1m 与官方日度 1m 归档逐位比对；
* ``maintenance_windows_vs_aggtrades`` 维护占位分钟内逐笔归档是否确无成交。
每个函数都返回所用归档的 URL 与 SHA256。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_1m as D
from .data_1m import INTERVAL_MS, PRICE_COLS, VALUE_COLS, agg_ohlcv, minute_grid
from .scope import DAY, IS_END, IS_START, MINUTE

VOL_ABS_TOL = 1e-6          # BTC：成交量绝对差阈值（归档成交量精度 0.001 BTC）


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


def _src(rec: dict) -> dict:
    return {k: rec.get(k) for k in ("url", "sha256", "verified_against", "rows", "agg_trades", "trade_id_gaps",
                                    "missing_trade_ids") if k in rec}


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
    ph_prev_close = None
    if len(ph):
        prev = grid["close"].shift(1)
        ph_prev_close = int((grid.loc[ph, "close"] == prev.loc[ph]).sum())
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
        "placeholder_minutes_priced_at_previous_close": ph_prev_close,
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

    价格要求逐位相等。成交量类字段按绝对差判定（成交量 / 主动买量 > 1e-6 BTC，笔数不等），并单独报告“参照为 0、
    1m 聚合为正”的 K 线（相对差为无穷大）、最大绝对差与最大相对差（参照为 0 时记为 inf）。
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
    out["ohlc_mismatch_bars"] = int(len(bad))
    out["ohlc_mismatch_bars_list"] = [str(t) for t in bad[:20]]
    for c in ("volume", "quote_volume", "taker_buy_volume", "count"):
        if c not in ref.columns:
            continue
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        ad = np.abs(x - y)
        tol = 0.5 if c == "count" else (VOL_ABS_TOL if c != "quote_volume" else 1e-2)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(y > 0, ad / np.where(y > 0, y, 1.0), np.where(ad > tol, np.inf, 0.0))
        out[f"{c}_mismatch_bars"] = int((ad > tol).sum())
        out[f"{c}_max_abs_diff"] = float(ad.max()) if len(ad) else 0.0
        out[f"{c}_max_rel_diff"] = float(rel.max()) if len(rel) else 0.0      # inf：参照为 0 而聚合为正
        out[f"{c}_reference_zero_aggregate_positive_bars"] = int(((y == 0) & (x > tol)).sum())
    vbad = common[(np.abs(a["volume"].to_numpy() - b["volume"].to_numpy()) > VOL_ABS_TOL)]
    out["volume_mismatch_bars_list"] = [[str(t), float(a.loc[t, "volume"]), float(b.loc[t, "volume"])] for t in vbad[:30]]
    if "raw_close_time_ms" in ref.columns:
        out["reference_close_time_mismatch_rows"] = int(
            (ref["raw_close_time_ms"].to_numpy() != _epoch_ms(ref.index) + INTERVAL_MS[rule.lower()] - 1).sum())
    return out


def daily_bars(df1m: pd.DataFrame) -> pd.DataFrame:
    """1m → UTC 日线（00:00 开盘），附 available_time = open_time + 24h（该日线完整收盘、可被使用的最早时刻）。"""
    d = agg_ohlcv(df1m, "1D")
    d["available_time"] = d.index + DAY
    return d


def qa_daily_available(df1m: pd.DataFrame, d1: pd.DataFrame, official_1d: pd.DataFrame) -> dict:
    """日线可用时间核验。available_time = open_time + 24h 在 daily_bars 中按定义成立（恒等式，不作为证据）；
    证据是：由 1m 数据独立求出每个自然日第一根与最后一根 1m 的开盘时间，核对第一根 = 日线 open_time、
    最后一根的收盘（开盘 + 1 分钟）= available_time，且每日恰 1,440 根；官方日线 close_time = open_time + 24h − 1ms。"""
    idx = pd.DatetimeIndex(d1.index).as_unit("ns")
    at = pd.DatetimeIndex(d1["available_time"]).as_unit("ns")
    t = pd.Series(pd.DatetimeIndex(df1m.index).as_unit("ns"), index=pd.DatetimeIndex(df1m.index).as_unit("ns"))
    day = t.dt.floor("D")
    first = t.groupby(day).min().reindex(idx)
    last = t.groupby(day).max().reindex(idx)
    out = {"days": int(len(d1)), "first_day": str(idx.min()), "last_day": str(idx.max()),
           "open_time_not_midnight_utc": int((idx.asi8 % (86_400 * 10**9) != 0).sum()),
           "identity_available_time_eq_open_plus_24h_by_construction": bool((at.asi8 - idx.asi8 == 86_400 * 10**9).all()),
           "days_first_1m_open_ne_day_open": int((first.to_numpy() != idx.to_numpy()).sum()),
           "days_last_1m_close_ne_available_time": int(((last + MINUTE).to_numpy() != at.to_numpy()).sum()),
           "days_with_ne_1440_minutes": int((d1["n_min"] != 1440).sum()),
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


# ----------------------------------------------------------------------------
# 逐笔归档定位
# ----------------------------------------------------------------------------
def tail_drop_analysis(trades: pd.DataFrame, minute: pd.Timestamp, kline: pd.Series) -> dict:
    """K 线是否恰为该分钟逐笔序列（按 agg_trade_id）的一个前缀：找累计成交量等于 K 线成交量的前缀长度 k，
    报告被截掉的尾部（笔数、成交量、在分钟内的毫秒区间、价格范围）以及前缀的 OHLC 是否与 K 线一致。"""
    m0 = int(minute.value // 1_000_000)
    tt = trades[(trades["transact_time"] >= m0) & (trades["transact_time"] < m0 + 60_000)]
    q = tt["quantity"].to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(q)])
    kv = float(kline["volume"])
    k = int(np.argmin(np.abs(cs - kv)))
    exact = bool(abs(cs[k] - kv) <= VOL_ABS_TOL)
    pre, tail = tt.iloc[:k], tt.iloc[k:]
    rec = {"minute": str(minute), "agg_trades_in_minute": int(len(tt)), "kline_volume": kv,
           "aggtrades_volume": float(q.sum()), "kline_is_exact_prefix": exact, "prefix_len": k,
           "prefix_volume_residual": float(cs[k] - kv)}
    if len(pre):
        rec["prefix_ohlc_equals_kline"] = bool(
            pre["price"].iloc[0] == kline["open"] and pre["price"].max() == kline["high"]
            and pre["price"].min() == kline["low"] and pre["price"].iloc[-1] == kline["close"])
    if len(tail):
        ms = tail["transact_time"].to_numpy() - m0
        rec.update({"dropped_agg_trades": int(len(tail)), "dropped_volume": float(tail["quantity"].sum()),
                    "dropped_ms_in_minute": [int(ms.min()), int(ms.max())],
                    "dropped_price_range": [float(tail["price"].min()), float(tail["price"].max())],
                    "true_close": float(tt["price"].iloc[-1]), "true_high": float(tt["price"].max()),
                    "true_low": float(tt["price"].min())})
    return rec


def localize_volume_discrepancies(df1m: pd.DataFrame, ref_1h: pd.DataFrame, official_1d: pd.DataFrame | None = None,
                                  tol_abs: float = VOL_ABS_TOL) -> dict:
    """1m 聚合与仓库 1H（逐小时）或官方 1d（逐日）成交量不一致的自然日：用该日官方 aggTrades 逐分钟定位差额。

    对每个自然日记录 1m / 1H / 官方 1d / 逐笔的全天成交量、逐笔归档相对 1m 的空洞分钟；逐笔多于 K 线 0.5 BTC 以上的分钟
    逐一做前缀（尾部截断）分析并分类（分钟末尾漏记 / 单笔跨分钟错位 / 其他）。只读数据，不改动 1m 缓存。
    """
    agg_h = agg_ohlcv(df1m, "1h")
    ch = agg_h.index.intersection(ref_1h.index)
    hours = ch[(np.abs(agg_h.loc[ch, "volume"].to_numpy() - ref_1h.loc[ch, "volume"].to_numpy()) > tol_abs)]
    days = {t.floor("D") for t in hours}
    days_1d = []
    if official_1d is not None:
        agg_d = agg_ohlcv(df1m, "1d")
        cd = agg_d.index.intersection(official_1d.index)
        d_bad = cd[(np.abs(agg_d.loc[cd, "volume"].to_numpy() - official_1d.loc[cd, "volume"].to_numpy()) > tol_abs)]
        days_1d = [str(t.date()) for t in d_bad]
        days |= set(d_bad)
    out = {"hours_flagged_vs_1h": [str(t) for t in hours], "days_flagged_vs_official_1d": days_1d, "days": [], "minutes": []}
    for day in sorted(days):
        trades, src = D.fetch_aggtrades(day)
        agg = D.aggtrades_to_1m(trades, day)
        kl = df1m[(df1m.index >= day) & (df1m.index < day + DAY)]
        h1d = ref_1h[(ref_1h.index >= day) & (ref_1h.index < day + DAY)]
        holes = D._agg_holes(kl, agg)
        vols = {"1m": float(kl["volume"].sum()), "repo_1h": float(h1d["volume"].sum()),
                "official_1d": float(official_1d.loc[day, "volume"]) if official_1d is not None and day in official_1d.index
                else None, "aggtrades": float(agg["volume"].sum())}
        out["days"].append({"day": str(day.date()), "aggtrades_archive": _src(src), "day_volume": vols,
                            "products_equal_to_aggtrades": [k for k, v in vols.items()
                                                            if v is not None and k != "aggtrades"
                                                            and abs(v - vols["aggtrades"]) <= tol_abs],
                            "aggtrades_hole_minutes_vs_1m": int(len(holes)),
                            "hours_flagged_vs_1h": [str(t) for t in hours if t.floor("D") == day]})
        dv = agg["volume"].reindex(kl.index).fillna(0.0) - kl["volume"]
        # 逐笔多于 K 线 0.5 BTC 以上的分钟做前缀分析并分类：
        #   tail_omission              K 线恰为（或残差 ≤ 0.5 BTC 地接近）该分钟逐笔序列的前缀，截掉 ≥ 2 笔，且收盘价因此不同；
        #   single_trade_displacement  只截掉 1 笔且其成交时刻在该分钟第 59 秒之后（这笔成交在 K 线中被归入其后的分钟）；
        #   other                      其余。
        # K 线多于逐笔 0.5 BTC 以上的分钟记为 kline_excess（被归入此处的错位成交，或自然日边界上的跨日归属差异）。
        v = dv.to_numpy()
        common = kl.index.intersection(agg.index)
        kc, ac = kl.loc[common], agg.loc[common]
        hlc = ((ac["high"] != kc["high"]) | (ac["low"] != kc["low"]) | (ac["close"] != kc["close"]))
        cls = {"tail_omission": [], "single_trade_displacement": [], "other": []}
        for t in kl.index[v > 0.5]:
            if t not in agg.index:
                continue
            a, k = agg.loc[t], kl.loc[t]
            td = tail_drop_analysis(trades, t, k)
            close_diff = bool(a["close"] != k["close"])
            near_prefix = td["kline_is_exact_prefix"] or abs(td["prefix_volume_residual"]) <= 0.5
            nd = td.get("dropped_agg_trades", 0)
            if near_prefix and nd >= 2 and close_diff:
                c = "tail_omission"
            elif nd == 1 and td["dropped_ms_in_minute"][0] >= 59_000:
                c = "single_trade_displacement"
            else:
                c = "other"
            cls[c].append(t)
            if c != "single_trade_displacement":
                out["minutes"].append({
                    "minute": str(t), "class": c, "volume_aggtrades_minus_1m": float(dv[t]),
                    "kline_hlc": [float(k["high"]), float(k["low"]), float(k["close"])],
                    "aggtrades_hlc": [float(a["high"]), float(a["low"]), float(a["close"])],
                    "extreme_differs": bool(a["high"] != k["high"] or a["low"] != k["low"]),
                    "close_differs": close_diff, "tail_drop": td})
        excess = kl.index[v < -0.5]
        in_cls = lambda c: common.isin(pd.DatetimeIndex(cls[c]))
        out["days"][-1].update({
            "net_volume_aggtrades_minus_1m": float(v.sum()),
            "tail_omission_minutes": [str(t) for t in cls["tail_omission"]],
            "tail_omission_volume": float(dv[cls["tail_omission"]].sum()) if cls["tail_omission"] else 0.0,
            "single_trade_displacement_minutes": len(cls["single_trade_displacement"]),
            "other_positive_minutes": [str(t) for t in cls["other"]],
            "kline_excess_minutes": [[str(t), float(dv[t])] for t in excess],
            "hlc_differs_minutes_total": int(hlc.sum()),
            "hlc_differs_in_tail_omission_minutes": int((hlc & in_cls("tail_omission")).sum())})
    return out


def monthly_vs_daily_check(df1m: pd.DataFrame, days: list[pd.Timestamp]) -> list[dict]:
    """指定自然日：缓存中的 1m（月度归档或已替换的日度归档）与官方日度 1m 归档逐位比对（全部数值列与原始收盘时间）。"""
    cols = VALUE_COLS + ["raw_close_time_ms"]
    out = []
    for day in sorted(days):
        daily, src = D.fetch_daily_1m(day)
        kl = df1m[(df1m.index >= day) & (df1m.index < day + DAY)]
        same = bool(len(kl) == len(daily) and kl.index.equals(daily.index)
                    and (kl[cols].to_numpy() == daily[cols].to_numpy()).all())
        out.append({"day": str(day.date()), "cache_src": sorted({int(x) for x in kl["src"]}) if "src" in kl else None,
                    "bitwise_equal": same, "daily_1m_archive": _src(src)})
    return out


def maintenance_windows_vs_aggtrades(df1m: pd.DataFrame) -> list[dict]:
    """维护占位分钟段：逐笔归档在该时段内是否确无成交，以及该日 1m 与逐笔的全天成交量。"""
    g = minute_grid(df1m)
    ph = g.index[g["placeholder"].to_numpy()]
    out = []
    for s, e, n in _ranges(ph, MINUTE):
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        day = s.floor("D")
        trades, src = D.fetch_aggtrades(day)
        s_ms, e_ms = int(s.value // 1_000_000), int((e + MINUTE).value // 1_000_000)
        inside = trades[(trades["transact_time"] >= s_ms) & (trades["transact_time"] < e_ms)]
        kl = df1m[(df1m.index >= day) & (df1m.index < day + DAY)]
        out.append({"window": [str(s), str(e), n], "aggtrades_archive": _src(src), "agg_trades_inside_window": int(len(inside)),
                    "day_volume_1m": float(kl["volume"].sum()), "day_volume_aggtrades": float(trades["quantity"].sum())})
    return out


def background_day(day: pd.Timestamp, df1m: pd.DataFrame) -> dict:
    """高波动对照日：缓存 1m 与官方日度 1m 是否逐位相同，以及 K 线与逐笔聚合的背景差异（全天合计、两个方向的缺失分钟）。"""
    rec = D._kline_vs_agg_background(day, df1m)
    rec["daily_1m_archive"] = _src(rec["daily_1m_archive"])
    rec["aggtrades_archive"] = _src(rec["aggtrades_archive"])
    return rec

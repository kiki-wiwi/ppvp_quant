"""
ppvp_audit/data_1m.py — IS 区间（2020-01 至 2023-12）BTCUSDT 永续 1 分钟 K 线：下载、校验、缓存与一致性核对。

用途：只用于 (a) 解析 1H K 线内入场、止损、止盈、POC 的先后（同根歧义），(b) 抽样核对“1H 振幅内均匀分配成交量”
的成交量分布近似，(c) C1 限价单的触价 / 穿价成交判定。1 分钟数据不改变基线策略的任何判定。

来源：Binance 官方公开归档 data.binance.vision 的 USDⓈ-M 永续月度 zip，逐文件用官方 .CHECKSUM 做 SHA256 校验
（复用 data_loader.fetch_vision_zip）。只下载 IS 月份，不触及 2024 年及之后的任何文件。
缓存：data/processed_1m/BTCUSDT_perp_1m_IS.parquet（约 2.1M 行，不入库，可由 data/manifest_1m_is.json 的
URL 与 SHA256 重新下载复现）；manifest 入库。
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from config import IS_END, IS_START
from data_loader import _vision_path, fetch_vision_zip, month_range, parse_kline_csv

ROOT = Path(__file__).resolve().parents[1]
PROC_1M = ROOT / "data" / "processed_1m" / "BTCUSDT_perp_1m_IS.parquet"
MANIFEST_1M = ROOT / "data" / "manifest_1m_is.json"
IS_MONTHS = ("2020-01", "2023-12")


def build_1m_cache(workers: int = 8) -> dict:
    """下载并校验 IS 的 48 个月度 1m zip，写入 Parquet 缓存与 manifest。"""
    paths = [_vision_path("futures/um", "klines", "BTCUSDT", ym, "1m") for ym in month_range(*IS_MONTHS)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(fetch_vision_zip, paths))
    frames, files = [], []
    for csv_bytes, src in results:
        f = parse_kline_csv(csv_bytes)
        src["rows"] = int(len(f))
        frames.append(f)
        files.append(src)
    df = pd.concat(frames).sort_index()
    if df.index.has_duplicates:
        raise ValueError("duplicate 1m open times")
    df = df[(df.index >= IS_START) & (df.index < IS_END)].drop(columns=["raw_close_time_ms"])
    df["src"] = np.int8(0)                                   # 0 = 月度归档，1 = 日度归档替换（见下）
    # 月度 1m 与研究用 1H 的 OHLC 不一致的自然日：用同日的官方日度 1m 归档替换，前提是日度归档聚合后与 1H 逐位一致
    # （Phase 0 记录的 2023-11-10 即如此：月度 1m 与官方 4H 归档在该日有误，日度 1m 与 1H 一致）。
    import pipeline
    df1h, _, _ = pipeline.load_market("BTCUSDT", IS_END)
    cols = ["open", "high", "low", "close"]
    bad_hours = _ohlc_mismatch_hours(df, df1h)
    replaced, unresolved = [], []
    for day in sorted({t.floor("D") for t in bad_hours}):
        rel = f"futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-{day:%Y-%m-%d}.zip"
        csv_bytes, src = fetch_vision_zip(rel)
        d = parse_kline_csv(csv_bytes).drop(columns=["raw_close_time_ms"])
        d = d[(d.index >= day) & (d.index < day + pd.Timedelta(days=1))]
        if len(d) == 1440 and not _ohlc_mismatch_hours(d, df1h.loc[(df1h.index >= day) & (df1h.index < day + pd.Timedelta(days=1))]):
            d["src"] = np.int8(1)
            df = pd.concat([df[(df.index < day) | (df.index >= day + pd.Timedelta(days=1))], d]).sort_index()
            replaced.append({"day": str(day.date()), "hours_fixed": [str(t) for t in bad_hours if t.floor("D") == day], **src})
        else:
            unresolved.append(str(day.date()))
    still_bad = _ohlc_mismatch_hours(df, df1h)
    PROC_1M.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROC_1M)
    import hashlib
    man = {"dataset": "BTCUSDT USDⓈ-M perp 1m klines, IS only", "window": [str(IS_START), str(IS_END)],
           "venue": "binance", "market_type": "usdm_perp", "provider": "binance_vision_futures_um",
           "is_missing": 0, "is_imputed": 0, "is_partial": 0,
           "rows": int(len(df)), "path": str(PROC_1M.relative_to(ROOT)),
           "parquet_sha256": hashlib.sha256(PROC_1M.read_bytes()).hexdigest(), "files": files,
           "daily_replacements": replaced, "unresolved_days": unresolved,
           "ohlc_mismatch_hours_after_replacement": [str(t) for t in still_bad],
           "note": "raw zips live under data/raw/ (git-ignored); the parquet is git-ignored and rebuilt by "
                   "python -m ppvp_audit.data_1m --build. Column src: 0 = monthly archive, 1 = daily archive."}
    MANIFEST_1M.write_text(json.dumps(man, indent=1, ensure_ascii=False))
    return man


def _ohlc_mismatch_hours(df1m: pd.DataFrame, df1h: pd.DataFrame) -> list:
    g = df1m.resample("1h", origin="epoch", label="left", closed="left")
    agg = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last()}).dropna()
    common = df1h.index.intersection(agg.index)
    a, b = agg.loc[common], df1h.loc[common]
    bad = (a["open"] != b["open"]) | (a["high"] != b["high"]) | (a["low"] != b["low"]) | (a["close"] != b["close"])
    return list(common[bad.to_numpy()])


def load_1m() -> pd.DataFrame:
    if not PROC_1M.exists():
        build_1m_cache()
    df = pd.read_parquet(PROC_1M)
    if len(df) and (df.index.min() < IS_START or df.index.max() >= IS_END):
        raise RuntimeError("1m cache extends outside IS")
    return df


def qa_1m_vs_1h(df1m: pd.DataFrame, df1h: pd.DataFrame) -> dict:
    """1m 聚合到 1H 后与研究用 1H 逐根比对：缺失分钟、OHLC 是否逐位一致、成交量相对差。"""
    g = df1m.resample("1h", origin="epoch", label="left", closed="left")
    agg = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum(), "n_min": g["close"].count()})
    agg = agg[agg["n_min"] > 0]
    common = df1h.index.intersection(agg.index)
    a, b = agg.loc[common], df1h.loc[common]
    rel_v = (a["volume"] - b["volume"]).abs() / b["volume"].where(b["volume"] > 0)
    out = {
        "hours_1h": int(len(df1h)), "hours_with_1m": int(len(agg)), "hours_common": int(len(common)),
        "hours_in_1h_without_1m": [str(t) for t in df1h.index.difference(agg.index)],
        "hours_with_missing_minutes": int((a["n_min"] < 60).sum()),
        "missing_minutes_total": int((60 - a["n_min"]).clip(lower=0).sum()),
        "open_mismatch": int((a["open"] != b["open"]).sum()), "high_mismatch": int((a["high"] != b["high"]).sum()),
        "low_mismatch": int((a["low"] != b["low"]).sum()), "close_mismatch": int((a["close"] != b["close"]).sum()),
        "volume_rel_diff_max": float(rel_v.max()), "volume_rel_diff_gt_1e-6": int((rel_v > 1e-6).sum()),
    }
    bad = common[(a["high"] != b["high"]) | (a["low"] != b["low"]) | (a["open"] != b["open"]) | (a["close"] != b["close"])]
    out["ohlc_mismatch_hours"] = [str(t) for t in bad[:50]]
    out["hours_with_missing_minutes_list"] = [str(t) for t in a.index[a["n_min"] < 60][:200]]
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if args.build:
        m = build_1m_cache()
        print(m["rows"], m["parquet_sha256"])

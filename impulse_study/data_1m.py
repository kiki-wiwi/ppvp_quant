"""
impulse_study/data_1m.py — Checkpoint 1 数据工程：BTCUSDT 永续 1m（IS）与预热数据的下载、校验、裁决与缓存。

来源与校验
----------
全部取自 Binance 官方公开归档 data.binance.vision，每个 zip 用官方 ``.CHECKSUM`` 做 SHA256 校验
（复用 ``data_loader.fetch_vision_zip``，原始 zip 缓存在 git 忽略的 ``data/raw/``）：

* 永续 1m 月度归档 2020-01 … 2023-12（48 个，主数据）；
* 永续 15m / 1d 月度归档 2020-01 … 2023-12（各 48 个，只作交叉对齐核验，不进入任何信号计算）；
* 现货 1d 月度归档 2019-09 … 2019-12（ATR_1D 的价格预热；Binance Vision 的永续归档始于 2020-01，
  2019 年没有永续数据可用）；
* 现货 1m 月度归档 2019-11 … 2019-12（只供“用现货成交量拼接预热”的计数敏感性 S4，不进入主口径）；
* 若某自然日的月度 1m 聚合与仓库 1H 不一致，再下载该日的官方日度 1m 归档与逐笔聚合成交（aggTrades）归档做三方裁决。

不下载 2024-01 及之后的任何文件，也不读取 ETH。

缓存
----
``data/processed_1m/*.parquet``（git 忽略，约 2.1M 行；由本模块 ``--build`` 按 manifest 中的 URL 与 SHA256 重建），
``data/manifest_impulse_1m.json``（入库）：逐文件 URL / SHA256 / 行数 / 表头格式、裁决证据、各缓存文件的
内容哈希（对索引与各列的 little-endian 数组求 SHA256，与 parquet 写出库版本无关）。

时间戳约定（与仓库一致）：索引 = UTC 开盘时间 open_time（ns 精度），K 线在 open_time + interval 收盘后可用；
原始 close_time 保留为 ``raw_close_time_ms`` 供 QA 核对 ``= open_time + interval − 1ms``。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import _has_header, _vision_path, fetch_vision_zip, month_range, parse_kline_csv

from .scope import (IS_END, IS_MONTHS, IS_START, MINUTE, SPOT_1D_WARMUP_MONTHS, SPOT_1M_WARMUP_MONTHS, SYMBOL,
                    assert_before_is, assert_within_is)

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "processed_1m"
MANIFEST_PATH = ROOT / "data" / "manifest_impulse_1m.json"
CACHE_FILES = {
    "perp_1m": "BTCUSDT_perp_1m_IS.parquet",
    "perp_15m_official": "BTCUSDT_perp_15m_official_IS.parquet",
    "perp_1d_official": "BTCUSDT_perp_1d_official_IS.parquet",
    "spot_1d_warmup": "BTCUSDT_spot_1d_warmup_2019.parquet",
    "spot_1m_warmup": "BTCUSDT_spot_1m_warmup_2019.parquet",
}
PRICE_COLS = ["open", "high", "low", "close"]
VALUE_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]
INTERVAL_MS = {"1m": 60_000, "15m": 900_000, "1h": 3_600_000, "1d": 86_400_000}


# ----------------------------------------------------------------------------
# 下载与解析
# ----------------------------------------------------------------------------
def _parse(csv_bytes: bytes) -> pd.DataFrame:
    df = parse_kline_csv(csv_bytes)
    df.index = pd.DatetimeIndex(df.index).as_unit("ns")
    return df


def fetch_monthly(market: str, interval: str, months: tuple[str, str], workers: int = 8) -> tuple[pd.DataFrame, list[dict]]:
    """下载并校验一段月度 K 线归档，返回 (按时间排序的 DataFrame, 逐文件来源记录)。"""
    paths = [_vision_path(market, "klines", SYMBOL, ym, interval) for ym in month_range(*months)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(fetch_vision_zip, paths))
    frames, files = [], []
    for csv_bytes, src in results:
        f = _parse(csv_bytes)
        files.append({**src, "rows": int(len(f)), "has_header": bool(_has_header(csv_bytes)),
                      "first_open_time": str(f.index.min()), "last_open_time": str(f.index.max())})
        frames.append(f)
    return pd.concat(frames).sort_index(kind="stable"), files


def fetch_daily_1m(day: pd.Timestamp, market: str = "futures/um") -> tuple[pd.DataFrame, dict]:
    rel = f"{market}/daily/klines/{SYMBOL}/1m/{SYMBOL}-1m-{day:%Y-%m-%d}.zip"
    csv_bytes, src = fetch_vision_zip(rel)
    d = _parse(csv_bytes)
    d = d[(d.index >= day) & (d.index < day + pd.Timedelta(days=1))]
    return d, {**src, "rows": int(len(d)), "has_header": bool(_has_header(csv_bytes))}


AGG_COLS_UM = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]


def fetch_aggtrades_1m(day: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """官方逐笔聚合成交归档 → 1m OHLCV（open/close 按 agg_trade_id 先后，taker 买量 = is_buyer_maker 为 False 的成交量）。"""
    rel = f"futures/um/daily/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{day:%Y-%m-%d}.zip"
    csv_bytes, src = fetch_vision_zip(rel)
    hdr = _has_header(csv_bytes)
    t = pd.read_csv(io.BytesIO(csv_bytes), header=0 if hdr else None)
    if t.shape[1] != len(AGG_COLS_UM):
        raise ValueError(f"unexpected aggTrades column count {t.shape[1]}")
    t.columns = AGG_COLS_UM
    t = t.sort_values("agg_trade_id", kind="stable")
    ibm = t["is_buyer_maker"]
    if ibm.dtype != bool:
        ibm = ibm.astype(str).str.lower().map({"true": True, "false": False})
    t["minute"] = pd.to_datetime(t["transact_time"].astype("int64") // 60_000 * 60_000, unit="ms", utc=True)
    t["quote"] = t["price"] * t["quantity"]
    t["n"] = t["last_trade_id"] - t["first_trade_id"] + 1
    t["taker_buy"] = np.where(ibm.to_numpy(dtype=bool), 0.0, t["quantity"].to_numpy())
    g = t.groupby("minute", sort=True)
    out = pd.DataFrame({"open": g["price"].first(), "high": g["price"].max(), "low": g["price"].min(),
                        "close": g["price"].last(), "volume": g["quantity"].sum(), "quote_volume": g["quote"].sum(),
                        "count": g["n"].sum().astype("float64"), "taker_buy_volume": g["taker_buy"].sum()})
    out.index = pd.DatetimeIndex(out.index).as_unit("ns")
    out.index.name = "open_time"
    out = out[(out.index >= day) & (out.index < day + pd.Timedelta(days=1))]
    return out, {**src, "agg_trades": int(len(t)), "has_header": bool(hdr), "minutes": int(len(out))}


# ----------------------------------------------------------------------------
# 1m 与仓库 1H 的逐小时比对、问题自然日的三方裁决
# ----------------------------------------------------------------------------
PANDAS_RULE = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "1d": "24h", "1D": "24h"}   # 24h + origin=epoch = UTC 00:00 边界


def agg_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """按 UTC 绝对边界聚合（origin=epoch，label=left，closed=left），n_min 记录参与聚合的行数。"""
    g = df.resample(PANDAS_RULE.get(rule, rule), origin="epoch", label="left", closed="left")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum(), "quote_volume": g["quote_volume"].sum(),
                        "count": g["count"].sum(), "taker_buy_volume": g["taker_buy_volume"].sum(),
                        "n_min": g["close"].count()})
    return out[out["n_min"] > 0]


def ohlc_mismatch_index(a: pd.DataFrame, b: pd.DataFrame) -> pd.DatetimeIndex:
    common = a.index.intersection(b.index)
    x, y = a.loc[common, PRICE_COLS].to_numpy(), b.loc[common, PRICE_COLS].to_numpy()
    return common[(x != y).any(axis=1)]


def _minute_diff(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    common = a.index.intersection(b.index)
    x, y = a.loc[common], b.loc[common]
    px = (x[PRICE_COLS].to_numpy() != y[PRICE_COLS].to_numpy()).any(axis=1)
    rv = (x["volume"] - y["volume"]).abs() / y["volume"].where(y["volume"] > 0)
    vbad = (rv > 1e-9).to_numpy()
    bad = common[px | vbad]
    return {"minutes_compared": int(len(common)), "only_in_first": int(len(a.index.difference(b.index))),
            "only_in_second": int(len(b.index.difference(a.index))), "ohlc_mismatch_minutes": int(px.sum()),
            "volume_mismatch_minutes_rel_gt_1e-9": int(vbad.sum()),
            "first_mismatch": str(bad.min()) if len(bad) else None, "last_mismatch": str(bad.max()) if len(bad) else None,
            "volume_sum_first": float(x["volume"].sum()), "volume_sum_second": float(y["volume"].sum())}


def _silent_traded_minutes(kl: pd.DataFrame, agg: pd.DataFrame) -> pd.DatetimeIndex:
    """K 线记为零成交（或缺失）、逐笔归档却有成交的分钟。"""
    traded = agg.index[agg["volume"] > 0]
    has_vol = kl.index[kl["volume"] > 0]
    return traded.difference(has_vol)


def _kline_vs_agg_background(day: pd.Timestamp, monthly_all: pd.DataFrame) -> dict:
    """对照日：月度与日度 1m 逐位相同的相邻自然日上，K 线与逐笔聚合之间的“背景”差异。"""
    daily, _ = fetch_daily_1m(day)
    agg, _ = fetch_aggtrades_1m(day)
    mon = monthly_all[(monthly_all.index >= day) & (monthly_all.index < day + pd.Timedelta(days=1))]
    same = len(mon) == len(daily) and bool((mon[VALUE_COLS].to_numpy() == daily[VALUE_COLS].to_numpy()).all())
    d = _minute_diff(daily[daily["volume"] > 0], agg)
    return {"day": str(day.date()), "monthly_equals_daily_bitwise": same,
            "kline_vs_aggtrades_ohlc_mismatch_minutes": d["ohlc_mismatch_minutes"],
            "kline_vs_aggtrades_volume_mismatch_minutes": d["volume_mismatch_minutes_rel_gt_1e-9"],
            "day_volume_kline": d["volume_sum_first"], "day_volume_aggtrades": d["volume_sum_second"],
            "silent_traded_minutes": int(len(_silent_traded_minutes(daily, agg)))}


def adjudicate_day(day: pd.Timestamp, monthly_day: pd.DataFrame, h1_day: pd.DataFrame,
                   monthly_all: pd.DataFrame | None = None) -> dict:
    """某自然日月度 1m 聚合与 1H 不一致：取官方日度 1m 与 aggTrades 逐笔聚合做三方比对。

    规则 v1（下载前写定）要求日度 1m 与 aggTrades 逐分钟 OHLC 与成交量完全一致。实测在月度 = 日度逐位相同的对照日上，
    K 线与逐笔聚合也有成百分钟不一致（几乎全部是 open 价，以及成交被归入相邻分钟，全天合计成交量逐位相等），
    即逐笔聚合本身不能逐分钟复原 K 线，v1 无法区分“归档缺陷”与“背景差异”。因此改用规则 v2（只涉及数据，不涉及任何
    策略结果）：日度 1m 整日替换月度 1m，当且仅当
      (a) 日度 1m 的 24 根 1H 聚合与仓库 1H 的 OHLC 逐位一致；
      (b) 日度 1m 的全天成交量与 aggTrades 全天成交量相对差 ≤ 1e-9；
      (c) 日度 1m 没有“记为零成交、逐笔归档却有成交”的分钟；
    且月度 1m 违反 (a) 或 (c)。两条规则的判定都写入记录；对照日的背景差异一并记录。
    """
    daily, dsrc = fetch_daily_1m(day)
    agg, asrc = fetch_aggtrades_1m(day)
    traded = lambda d: d[d["volume"] > 0]          # 无成交的分钟在逐笔归档中没有记录
    silent_m = _silent_traded_minutes(monthly_day, agg)
    silent_d = _silent_traded_minutes(daily, agg)
    rec = {"day": str(day.date()), "daily_1m_archive": dsrc, "aggtrades_archive": asrc,
           "zero_volume_minutes": {"monthly": int((monthly_day["volume"] == 0).sum()),
                                   "daily": int((daily["volume"] == 0).sum())},
           "silent_traded_minutes": {"monthly": int(len(silent_m)), "daily": int(len(silent_d)),
                                     "monthly_range": [str(silent_m.min()), str(silent_m.max())] if len(silent_m) else None,
                                     "aggtrades_volume_in_monthly_silent_minutes":
                                         float(agg.loc[silent_m, "volume"].sum()) if len(silent_m) else 0.0},
           "monthly_vs_daily": _minute_diff(monthly_day, daily),
           "monthly_vs_aggtrades": _minute_diff(traded(monthly_day), agg),
           "daily_vs_aggtrades": _minute_diff(traded(daily), agg),
           "day_volume": {"monthly": float(monthly_day["volume"].sum()), "daily": float(daily["volume"].sum()),
                          "aggtrades": float(agg["volume"].sum())},
           "hours_mismatch_monthly_vs_1h": [str(t) for t in ohlc_mismatch_index(agg_ohlcv(monthly_day, "1h"), h1_day)],
           "hours_mismatch_daily_vs_1h": [str(t) for t in ohlc_mismatch_index(agg_ohlcv(daily, "1h"), h1_day)],
           "hours_mismatch_aggtrades_vs_1h": [str(t) for t in ohlc_mismatch_index(agg_ohlcv(agg, "1h"), h1_day)]}
    dv = rec["daily_vs_aggtrades"]
    rec["rule_v1_strict_pass"] = bool(len(daily) == 1440 and dv["ohlc_mismatch_minutes"] == 0
                                      and dv["volume_mismatch_minutes_rel_gt_1e-9"] == 0 and dv["only_in_first"] == 0
                                      and not rec["hours_mismatch_daily_vs_1h"])
    vol_ok = abs(rec["day_volume"]["daily"] - rec["day_volume"]["aggtrades"]) <= 1e-9 * rec["day_volume"]["aggtrades"]
    daily_ok = len(daily) == 1440 and not rec["hours_mismatch_daily_vs_1h"] and vol_ok and not len(silent_d)
    monthly_bad = bool(rec["hours_mismatch_monthly_vs_1h"]) or bool(len(silent_m))
    rec["rule_v2"] = {"a_daily_1h_ohlc_equal": not rec["hours_mismatch_daily_vs_1h"], "b_daily_day_volume_equals_aggtrades": vol_ok,
                      "c_daily_no_silent_traded_minutes": not len(silent_d), "monthly_violates_a_or_c": monthly_bad}
    if monthly_all is not None:
        rec["control_days_background"] = [
            _kline_vs_agg_background(c, monthly_all) for c in (day - pd.Timedelta(days=1), day + pd.Timedelta(days=1))
            if IS_START <= c < IS_END]
    rec["verdict"] = "replace_with_daily_archive" if (daily_ok and monthly_bad) else "unresolved_keep_monthly"
    rec["_daily_df"] = daily
    return rec


# ----------------------------------------------------------------------------
# 内容哈希、读取入口
# ----------------------------------------------------------------------------
def content_sha256(df: pd.DataFrame) -> str:
    """索引（int64 ns）与各列（按列名排序，float→<f8、int→<i8、bool→u1）的 SHA256，与 parquet 写出版本无关。"""
    h = hashlib.sha256()
    h.update(np.asarray(pd.DatetimeIndex(df.index).as_unit("ns").asi8, dtype="<i8").tobytes())
    for c in sorted(df.columns):
        a = df[c].to_numpy()
        if a.dtype.kind == "f":
            a = a.astype("<f8")
        elif a.dtype.kind in "iu":
            a = a.astype("<i8")
        elif a.dtype.kind == "b":
            a = a.astype("u1")
        else:
            raise TypeError(f"column {c} has unsupported dtype {a.dtype}")
        h.update(c.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _write(df: pd.DataFrame, key: str) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / CACHE_FILES[key]
    df.to_parquet(p, compression="zstd")
    return {"path": str(p.relative_to(ROOT)), "rows": int(len(df)), "first_open_time": str(df.index.min()),
            "last_open_time": str(df.index.max()), "content_sha256": content_sha256(df),
            "parquet_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def _manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} missing; build with `python -m impulse_study.data_1m --build`")
    return json.loads(MANIFEST_PATH.read_text())


def _load(key: str, verify: bool) -> pd.DataFrame:
    p = CACHE_DIR / CACHE_FILES[key]
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; rebuild with `python -m impulse_study.data_1m --build` "
                                "(downloads the official archives listed in the manifest)")
    df = pd.read_parquet(p)
    df.index = pd.DatetimeIndex(df.index).as_unit("ns")
    if verify:
        want = _manifest()["caches"][key]["content_sha256"]
        got = content_sha256(df)
        if got != want:
            raise ValueError(f"{p}: content hash {got} differs from manifest {want}")
    return df


def load_perp_1m(verify: bool = True) -> pd.DataFrame:
    """IS 永续 1m（月度归档 + 经裁决的日度替换；列 src：0 = 月度，1 = 日度替换）。不含缺失分钟的占位行。"""
    df = _load("perp_1m", verify)
    assert_within_is(df.index, "perp 1m")
    return df


def load_official(interval: str, verify: bool = True) -> pd.DataFrame:
    """官方永续 15m / 1d 归档（IS，仅供交叉核验）。"""
    df = _load({"15m": "perp_15m_official", "1d": "perp_1d_official"}[interval], verify)
    assert_within_is(df.index, f"official perp {interval}")
    return df


def load_spot_warmup(interval: str, verify: bool = True) -> pd.DataFrame:
    """2020 年之前的现货预热数据（1d：ATR_1D 价格预热；1m：只供 S4 计数敏感性）。"""
    df = _load({"1d": "spot_1d_warmup", "1m": "spot_1m_warmup"}[interval], verify)
    assert_before_is(df.index, f"spot {interval} warm-up")
    return df


def minute_grid(df1m: pd.DataFrame, start: pd.Timestamp = IS_START, end: pd.Timestamp = IS_END) -> pd.DataFrame:
    """把 1m 数据对齐到完整的 UTC 分钟网格 [start, end)，并加入逐分钟标记：

    missing      归档中不存在该分钟（价格列为 NaN）；
    zero_volume  成交量为 0；
    placeholder  成交量为 0 且 O = H = L = C（交易所维护 / 停机期间的占位 K 线）；
    bad          missing 或 placeholder：该分钟的价格不作为任何收益率的端点。
    """
    grid = pd.date_range(start, end - MINUTE, freq="1min", tz="UTC").as_unit("ns")
    out = df1m.reindex(grid)
    out.index.name = "open_time"
    out["missing"] = out["close"].isna().to_numpy()
    vol0 = (out["volume"] == 0).to_numpy()
    flat = ((out["open"] == out["high"]) & (out["high"] == out["low"]) & (out["low"] == out["close"])).to_numpy()
    out["zero_volume"] = vol0
    out["placeholder"] = vol0 & flat
    out["bad"] = out["missing"].to_numpy() | out["placeholder"].to_numpy()
    return out


# ----------------------------------------------------------------------------
# 构建
# ----------------------------------------------------------------------------
def build(workers: int = 8) -> dict:
    from ppvp_audit.common import load_is_market   # 上一轮的 IS 截断入口（含越界断言）

    df1h, _, _ = load_is_market()
    df1h = df1h.copy()
    df1h.index = pd.DatetimeIndex(df1h.index).as_unit("ns")

    print("[perp 1m] monthly archives 2020-01 .. 2023-12", flush=True)
    perp_raw, files_1m = fetch_monthly("futures/um", "1m", IS_MONTHS, workers)
    raw_stats = {"rows_downloaded": int(len(perp_raw)), "duplicate_open_times": int(perp_raw.index.duplicated().sum()),
                 "rows_outside_is": int(((perp_raw.index < IS_START) | (perp_raw.index >= IS_END)).sum())}
    if raw_stats["duplicate_open_times"]:
        raise ValueError("duplicate 1m open times in the monthly archives")
    perp = perp_raw[(perp_raw.index >= IS_START) & (perp_raw.index < IS_END)].copy()
    perp["src"] = np.int8(0)

    bad_hours = ohlc_mismatch_index(agg_ohlcv(perp, "1h"), df1h)
    adjud = []
    for day in sorted({t.floor("D") for t in bad_hours}):
        print(f"[perp 1m] adjudicating {day.date()} (monthly 1m vs 1H mismatch)", flush=True)
        mday = perp[(perp.index >= day) & (perp.index < day + pd.Timedelta(days=1))]
        hday = df1h[(df1h.index >= day) & (df1h.index < day + pd.Timedelta(days=1))]
        rec = adjudicate_day(day, mday, hday, monthly_all=perp)
        daily = rec.pop("_daily_df")
        if rec["verdict"] == "replace_with_daily_archive":
            daily = daily.copy()
            daily["src"] = np.int8(1)
            keep = (perp.index < day) | (perp.index >= day + pd.Timedelta(days=1))
            perp = pd.concat([perp[keep], daily[perp.columns]]).sort_index(kind="stable")
        adjud.append(rec)
    still_bad = ohlc_mismatch_index(agg_ohlcv(perp, "1h"), df1h)
    assert_within_is(perp.index, "perp 1m")

    print("[perp 15m / 1d] official archives (cross-check only)", flush=True)
    p15, files_15 = fetch_monthly("futures/um", "15m", IS_MONTHS, workers)
    p1d, files_1d = fetch_monthly("futures/um", "1d", IS_MONTHS, workers)
    p15 = p15[(p15.index >= IS_START) & (p15.index < IS_END)]
    p1d = p1d[(p1d.index >= IS_START) & (p1d.index < IS_END)]

    print("[spot warm-up] 1d 2019-09 .. 2019-12, 1m 2019-11 .. 2019-12", flush=True)
    s1d, files_s1d = fetch_monthly("spot", "1d", SPOT_1D_WARMUP_MONTHS, workers)
    s1m, files_s1m = fetch_monthly("spot", "1m", SPOT_1M_WARMUP_MONTHS, workers)
    s1d, s1m = s1d[s1d.index < IS_START], s1m[s1m.index < IS_START]
    assert_before_is(s1d.index, "spot 1d warm-up")
    assert_before_is(s1m.index, "spot 1m warm-up")

    caches = {"perp_1m": _write(perp, "perp_1m"), "perp_15m_official": _write(p15, "perp_15m_official"),
              "perp_1d_official": _write(p1d, "perp_1d_official"), "spot_1d_warmup": _write(s1d, "spot_1d_warmup"),
              "spot_1m_warmup": _write(s1m, "spot_1m_warmup")}
    man = {
        "dataset": "BTCUSDT USDⓈ-M perpetual 1m klines for the impulse study (IS only) + cross-check and warm-up archives",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "scope": {"symbol": SYMBOL, "is_window": [str(IS_START), str(IS_END)],
                  "perp_months": list(IS_MONTHS), "spot_1d_warmup_months": list(SPOT_1D_WARMUP_MONTHS),
                  "spot_1m_warmup_months": list(SPOT_1M_WARMUP_MONTHS),
                  "not_downloaded": "any file dated 2024-01 or later; any ETH file; perpetual archives before 2020-01 "
                                    "do not exist on data.binance.vision (HTTP 404 for 2019-08 .. 2019-12)"},
        "timestamp_convention": "index = UTC open_time (ns); a bar is usable after open_time + interval; "
                                "raw_close_time_ms = Binance close_time (open_time + interval - 1 ms)",
        "provider": "binance_vision (official monthly archives, SHA256 checked against the published .CHECKSUM)",
        "raw_download_stats_perp_1m": raw_stats,
        "files": {"perp_1m": files_1m, "perp_15m_official": files_15, "perp_1d_official": files_1d,
                  "spot_1d_warmup": files_s1d, "spot_1m_warmup": files_s1m},
        "adjudication_rule": "v2 (v1 recorded per day as rule_v1_strict_pass; see impulse_study.data_1m.adjudicate_day): "
                             "a calendar day whose monthly-1m hourly aggregate differs from the repository 1H OHLC is "
                             "replaced by the official daily 1m archive iff (a) the daily archive's 24 hourly aggregates "
                             "equal the repository 1H OHLC exactly, (b) its day volume equals the aggTrades day volume "
                             "(rel. <= 1e-9), (c) it has no zero-volume minute in which aggTrades records trades, and the "
                             "monthly archive violates (a) or (c). v1 required minute-exact equality with aggTrades, which "
                             "fails even on control days where the monthly and daily archives are bit-identical",
        "adjudicated_days": adjud,
        "hours_ohlc_mismatch_vs_1h_after_adjudication": [str(t) for t in still_bad],
        "caches": caches,
        "column_src": "perp_1m.src: 0 = monthly archive, 1 = official daily archive (adjudicated replacement)",
        "rebuild": "python -m impulse_study.data_1m --build   (raw zips under data/raw/, parquet under data/processed_1m/, "
                   "both git-ignored)",
    }
    MANIFEST_PATH.write_text(json.dumps(man, indent=1, ensure_ascii=False))
    return man


def verify() -> dict:
    """核对各缓存文件的内容哈希与 manifest 一致。"""
    man = _manifest()
    out = {}
    for key in CACHE_FILES:
        try:
            df = _load(key, verify=False)
            out[key] = content_sha256(df) == man["caches"][key]["content_sha256"]
        except FileNotFoundError:
            out[key] = "missing"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build / verify the impulse-study 1m cache (BTC IS only).")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.build:
        m = build()
        print(json.dumps({k: v for k, v in m.items() if k not in ("files",)}, indent=1, ensure_ascii=False)[:6000])
    if a.verify:
        print(json.dumps(verify(), indent=1))
    if not (a.build or a.verify):
        raise SystemExit("nothing to do: pass --build or --verify")

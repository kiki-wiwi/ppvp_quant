"""
impulse_study/data_1m.py — Checkpoint 1 数据工程：BTCUSDT 永续 1m（IS）与预热数据的下载、校验、裁决与缓存。

来源与校验
----------
全部取自 Binance 官方公开归档 data.binance.vision（原始 zip 缓存在 git 忽略的 ``data/raw/``）：

* 永续 1m 月度归档 2020-01 … 2023-12（48 个，主数据）；
* 永续 15m / 1d 月度归档 2020-01 … 2023-12（各 48 个，只作交叉对齐核验，不进入任何信号计算）；
* 现货 1d 月度归档 2019-09 … 2019-12（ATR_1D 的价格预热；Binance Vision 的永续归档始于 2020-01，
  2019 年没有永续数据可用）；
* 现货 1m 月度归档 2019-11 … 2019-12（只供“用现货成交量拼接预热”的计数敏感性 S4，不进入主口径）；
* 若某自然日的月度 1m 聚合与仓库 1H 不一致，再下载该日的官方日度 1m 归档与逐笔聚合成交（aggTrades）归档做三方裁决；
  QA（qa.py）另外下载若干自然日的日度 1m 与 aggTrades 归档做定位与核对，其来源记录写入 QA 结果。

每个下载入口都有范围断言：只允许 BTCUSDT；永续月度归档只允许 2020-01 … 2023-12，日度归档只允许 IS 内的自然日；
现货只允许上述预热月份。不下载 2024-01 及之后的任何文件，也不读取 ETH。

两种构建方式
------------
``--build``    首次构建：逐文件用官方 ``.CHECKSUM`` 做 SHA256 校验（data_loader.fetch_vision_zip），写缓存与 manifest。
               manifest 已存在时拒绝覆盖（除非加 ``--force-new-manifest``），覆盖时报告哪些内容哈希发生了变化。
``--rebuild``  按 manifest 复现：逐文件要求 SHA256 等于 manifest 中记录的值（不依赖在线 CHECKSUM），本地 ``data/raw/``
               有文件时可完全离线（``--offline``）；重建后的内容哈希必须与 manifest 逐一相等才写缓存；从不改写 manifest。
``--verify``   核对现有缓存的内容哈希与 manifest 一致。

缓存：``data/processed_1m/*.parquet``（git 忽略，约 2.1M 行）；``data/manifest_impulse_1m.json``（入库）记录逐文件
URL / SHA256 / 行数 / 表头格式、裁决证据（含对照日归档的来源）与各缓存的内容哈希（对索引与各列的 little-endian
数组求 SHA256，与 parquet 写出库的版本无关）。

时间戳约定（与仓库一致）：索引 = UTC 开盘时间 open_time（ns 精度），K 线在 open_time + interval 收盘后可用；
原始 close_time 保留为 ``raw_close_time_ms`` 供 QA 核对 ``= open_time + interval − 1ms``。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import RAW_DIR, VISION_BASE, _has_header, _http_get, _vision_path, fetch_vision_zip, month_range, \
    parse_kline_csv

from .scope import (IS_END, IS_MONTHS, IS_START, MINUTE, SPOT_1D_WARMUP_MONTHS, SPOT_1M_WARMUP_MONTHS, SYMBOL,
                    ScopeError, assert_before_is, assert_within_is)

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "processed_1m"
MANIFEST_PATH = ROOT / "data" / "manifest_impulse_1m.json"
REPO_1H_PATH = ROOT / "data" / "processed" / "BTCUSDT_perp_1h.parquet"
REPO_SPOT_1H_PATH = ROOT / "data" / "processed" / "BTCUSDT_spot_1h.parquet"
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
OFFLINE = os.environ.get("IMPULSE_OFFLINE", "0") == "1"      # 为 1 时任何网络访问都抛错（离线复现核验用）


# ----------------------------------------------------------------------------
# 范围断言（每个下载入口）
# ----------------------------------------------------------------------------
ALLOWED_MONTHS = {
    ("futures/um", "1m"): IS_MONTHS, ("futures/um", "15m"): IS_MONTHS, ("futures/um", "1d"): IS_MONTHS,
    ("spot", "1d"): SPOT_1D_WARMUP_MONTHS, ("spot", "1m"): SPOT_1M_WARMUP_MONTHS,
}


def _guard_months(market: str, interval: str, months: tuple[str, str]) -> None:
    allowed = ALLOWED_MONTHS.get((market, interval))
    if allowed is None:
        raise ScopeError(f"archive family {market}/{interval} is outside this round's scope")
    a, b = set(month_range(*allowed)), month_range(*months)
    bad = [m for m in b if m not in a]
    if bad:
        raise ScopeError(f"{market}/{interval} months {bad} are outside the allowed range {allowed}")


def _guard_day(day: pd.Timestamp) -> None:
    if not (IS_START <= day < IS_END) or day != day.floor("D"):
        raise ScopeError(f"daily archive {day} is outside the IS window [{IS_START}, {IS_END})")


def _guard_rel_path(rel: str) -> None:
    """manifest 中记录的路径同样只能指向 BTCUSDT 的允许月份 / 自然日。"""
    if f"/{SYMBOL}/" not in rel and f"/{SYMBOL}-" not in rel:
        raise ScopeError(f"{rel}: symbol outside scope")
    stem = rel.rsplit("/", 1)[1].removesuffix(".zip")
    date = stem.rsplit("-", 3)
    if "/monthly/" in rel:
        ym = "-".join(stem.split("-")[-2:])
        market = "spot" if rel.startswith("spot/") else "futures/um"
        interval = stem.split("-")[1]
        _guard_months(market, interval, (ym, ym))
    elif "/daily/" in rel:
        _guard_day(pd.Timestamp("-".join(date[-3:]), tz="UTC"))
    else:
        raise ScopeError(f"{rel}: unknown archive layout")


# ----------------------------------------------------------------------------
# 下载与解析
# ----------------------------------------------------------------------------
def _unzip_single(blob: bytes, url: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
        if len(names) != 1:
            raise ValueError(f"unexpected zip content {names} in {url}")
        return z.read(names[0])


def fetch_archive(rel: str, pinned_sha256: str | None = None) -> tuple[bytes, dict]:
    """取一个归档的 CSV 字节。

    pinned_sha256 为 None：首次构建，用官方在线 .CHECKSUM 校验（data_loader.fetch_vision_zip）。
    pinned_sha256 给定：只接受 SHA256 等于该值的字节；本地 data/raw/ 有该文件时不联网，否则下载后核对。
    """
    _guard_rel_path(rel)
    url = f"{VISION_BASE}/{rel}"
    if pinned_sha256 is None:
        if OFFLINE:
            raise RuntimeError(f"offline mode: cannot verify {url} against the live CHECKSUM")
        csv_bytes, src = fetch_vision_zip(rel)
        return csv_bytes, {**src, "verified_against": "live_official_CHECKSUM"}
    local = RAW_DIR / rel
    if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest() == pinned_sha256:
        blob, how = local.read_bytes(), "pinned_manifest_sha256_local_raw"
    else:
        if OFFLINE:
            raise RuntimeError(f"offline mode: {local} missing or does not match the pinned SHA256")
        blob = _http_get(url).content
        got = hashlib.sha256(blob).hexdigest()
        if got != pinned_sha256:
            raise ValueError(f"SHA256 mismatch for {url}: pinned {pinned_sha256}, downloaded {got}")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(blob)
        how = "pinned_manifest_sha256_downloaded"
    return _unzip_single(blob, url), {"url": url, "sha256": pinned_sha256, "checksum_verified": True,
                                      "verified_against": how}


def _parse(csv_bytes: bytes) -> pd.DataFrame:
    df = parse_kline_csv(csv_bytes)
    df.index = pd.DatetimeIndex(df.index).as_unit("ns")
    return df


def _file_record(csv_bytes: bytes, src: dict, f: pd.DataFrame) -> dict:
    return {**src, "rows": int(len(f)), "has_header": bool(_has_header(csv_bytes)),
            "first_open_time": str(f.index.min()), "last_open_time": str(f.index.max())}


def fetch_monthly(market: str, interval: str, months: tuple[str, str], workers: int = 8) -> tuple[pd.DataFrame, list[dict]]:
    """下载并校验一段月度 K 线归档，返回 (按时间排序的 DataFrame, 逐文件来源记录)。"""
    _guard_months(market, interval, months)
    paths = [_vision_path(market, "klines", SYMBOL, ym, interval) for ym in month_range(*months)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(fetch_archive, paths))
    frames, files = [], []
    for csv_bytes, src in results:
        f = _parse(csv_bytes)
        files.append(_file_record(csv_bytes, src, f))
        frames.append(f)
    return pd.concat(frames).sort_index(kind="stable"), files


def fetch_daily_1m(day: pd.Timestamp, market: str = "futures/um", pinned_sha256: str | None = None) -> tuple[pd.DataFrame, dict]:
    _guard_day(day)
    rel = f"{market}/daily/klines/{SYMBOL}/1m/{SYMBOL}-1m-{day:%Y-%m-%d}.zip"
    csv_bytes, src = fetch_archive(rel, pinned_sha256)
    d = _parse(csv_bytes)
    d = d[(d.index >= day) & (d.index < day + pd.Timedelta(days=1))]
    return d, {**src, "rows": int(len(d)), "has_header": bool(_has_header(csv_bytes))}


AGG_COLS_UM = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]


def fetch_aggtrades(day: pd.Timestamp, pinned_sha256: str | None = None) -> tuple[pd.DataFrame, dict]:
    """官方逐笔聚合成交归档（按 agg_trade_id 排序；is_buyer_maker 为 bool）。来源记录含成交 id 序列的缺口统计。"""
    _guard_day(day)
    rel = f"futures/um/daily/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{day:%Y-%m-%d}.zip"
    csv_bytes, src = fetch_archive(rel, pinned_sha256)
    hdr = _has_header(csv_bytes)
    t = pd.read_csv(io.BytesIO(csv_bytes), header=0 if hdr else None)
    if t.shape[1] != len(AGG_COLS_UM):
        raise ValueError(f"unexpected aggTrades column count {t.shape[1]}")
    t.columns = AGG_COLS_UM
    t = t.sort_values("agg_trade_id", kind="stable").reset_index(drop=True)
    ibm = t["is_buyer_maker"]
    if ibm.dtype != bool:
        ibm = ibm.astype(str).str.lower().map({"true": True, "false": False})
    t["is_buyer_maker"] = ibm.to_numpy(dtype=bool)
    gaps = t["first_trade_id"].to_numpy()[1:] - t["last_trade_id"].to_numpy()[:-1] - 1
    return t, {**src, "agg_trades": int(len(t)), "has_header": bool(hdr),
               "trade_id_gaps": int((gaps > 0).sum()), "missing_trade_ids": int(gaps[gaps > 0].sum())}


def aggtrades_to_1m(t: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """逐笔 → 1m OHLCV（open/close 按 agg_trade_id 先后；taker 买量 = is_buyer_maker 为 False 的成交量）。"""
    minute = pd.to_datetime(t["transact_time"].astype("int64") // 60_000 * 60_000, unit="ms", utc=True)
    x = pd.DataFrame({"minute": minute, "price": t["price"], "quantity": t["quantity"],
                      "quote": t["price"] * t["quantity"], "n": t["last_trade_id"] - t["first_trade_id"] + 1,
                      "taker_buy": np.where(t["is_buyer_maker"].to_numpy(dtype=bool), 0.0, t["quantity"].to_numpy())})
    g = x.groupby("minute", sort=True)
    out = pd.DataFrame({"open": g["price"].first(), "high": g["price"].max(), "low": g["price"].min(),
                        "close": g["price"].last(), "volume": g["quantity"].sum(), "quote_volume": g["quote"].sum(),
                        "count": g["n"].sum().astype("float64"), "taker_buy_volume": g["taker_buy"].sum()})
    out.index = pd.DatetimeIndex(out.index).as_unit("ns")
    out.index.name = "open_time"
    return out[(out.index >= day) & (out.index < day + pd.Timedelta(days=1))]


def fetch_aggtrades_1m(day: pd.Timestamp, pinned_sha256: str | None = None) -> tuple[pd.DataFrame, dict]:
    t, src = fetch_aggtrades(day, pinned_sha256)
    out = aggtrades_to_1m(t, day)
    return out, {**src, "minutes": int(len(out))}


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


def _ranges(ts: pd.DatetimeIndex) -> list[list]:
    out = []
    for t in ts:
        if out and t - pd.Timestamp(out[-1][1]) == MINUTE:
            out[-1][1], out[-1][2] = str(t), out[-1][2] + 1
        else:
            out.append([str(t), str(t), 1])
    return out


def _minute_diff(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """两份分钟数据的逐分钟比对。成交量差按绝对值（> 1e-6 BTC）判定，参照为 0 的分钟不会被漏掉；
    全天合计分别对各自的全部分钟求和（不只对共同分钟）。"""
    common = a.index.intersection(b.index)
    x, y = a.loc[common], b.loc[common]
    px = (x[PRICE_COLS].to_numpy() != y[PRICE_COLS].to_numpy()).any(axis=1)
    vbad = ((x["volume"] - y["volume"]).abs() > 1e-6).to_numpy()
    bad = common[px | vbad]
    return {"minutes_compared": int(len(common)), "only_in_first": int(len(a.index.difference(b.index))),
            "only_in_second": int(len(b.index.difference(a.index))), "ohlc_mismatch_minutes": int(px.sum()),
            "volume_mismatch_minutes_abs_gt_1e-6": int(vbad.sum()),
            "first_mismatch": str(bad.min()) if len(bad) else None, "last_mismatch": str(bad.max()) if len(bad) else None,
            "volume_sum_common_first": float(x["volume"].sum()), "volume_sum_common_second": float(y["volume"].sum()),
            "volume_sum_all_first": float(a["volume"].sum()), "volume_sum_all_second": float(b["volume"].sum())}


def _silent_traded_minutes(kl: pd.DataFrame, agg: pd.DataFrame) -> pd.DatetimeIndex:
    """K 线记为零成交（或缺失）、逐笔归档却有成交的分钟。"""
    traded = agg.index[agg["volume"] > 0]
    has_vol = kl.index[kl["volume"] > 0]
    return traded.difference(has_vol)


def _agg_holes(kl: pd.DataFrame, agg: pd.DataFrame) -> pd.DatetimeIndex:
    """K 线有成交、逐笔归档却没有任何记录的分钟（逐笔归档的空洞）。"""
    return kl.index[kl["volume"] > 0].difference(agg.index[agg["volume"] > 0])


def kline_vs_aggtrades_day(day: pd.Timestamp, kline_day: pd.DataFrame | None = None) -> dict:
    """某自然日官方日度 1m（或给定的 K 线）与逐笔聚合的逐分钟比对，含全天合计、两个方向的缺失分钟、逐笔 id 缺口与来源。"""
    daily, dsrc = fetch_daily_1m(day)
    agg, asrc = fetch_aggtrades_1m(day)
    kl = daily if kline_day is None else kline_day
    d = _minute_diff(kl[kl["volume"] > 0], agg)
    holes = _agg_holes(kl, agg)
    silent = _silent_traded_minutes(kl, agg)
    return {"day": str(day.date()), "daily_1m_archive": dsrc, "aggtrades_archive": asrc,
            "kline_vs_aggtrades_ohlc_mismatch_minutes": d["ohlc_mismatch_minutes"],
            "kline_vs_aggtrades_volume_mismatch_minutes": d["volume_mismatch_minutes_abs_gt_1e-6"],
            "day_volume_kline": float(kl["volume"].sum()), "day_volume_aggtrades": float(agg["volume"].sum()),
            "kline_traded_minutes_absent_in_aggtrades": int(len(holes)),
            "kline_volume_in_aggtrades_holes": float(kl.loc[holes, "volume"].sum()) if len(holes) else 0.0,
            "aggtrades_hole_ranges": _ranges(holes), "silent_traded_minutes": int(len(silent)),
            "_daily_df": daily, "_agg_df": agg}


def _kline_vs_agg_background(day: pd.Timestamp, monthly_all: pd.DataFrame) -> dict:
    """对照日：月度与日度 1m 逐位相同的相邻自然日上，K 线与逐笔聚合之间的“背景”差异（含来源记录）。"""
    rec = kline_vs_aggtrades_day(day)
    daily = rec.pop("_daily_df")
    rec.pop("_agg_df")
    mon = monthly_all[(monthly_all.index >= day) & (monthly_all.index < day + pd.Timedelta(days=1))]
    cols = VALUE_COLS + ["raw_close_time_ms"]
    rec["monthly_equals_daily_bitwise"] = bool(len(mon) == len(daily) and mon.index.equals(daily.index)
                                               and (mon[cols].to_numpy() == daily[cols].to_numpy()).all())
    return rec


def adjudicate_day(day: pd.Timestamp, monthly_day: pd.DataFrame, h1_day: pd.DataFrame,
                   monthly_all: pd.DataFrame | None = None) -> dict:
    """某自然日月度 1m 聚合与 1H 不一致：取官方日度 1m 与 aggTrades 逐笔聚合做三方比对。

    规则 v1（首次运行构建前写入代码，同一会话内改写，未单独提交，故无法从 git 历史核实先后）要求日度 1m 与 aggTrades
    逐分钟 OHLC 与成交量完全一致。实测在月度 = 日度逐位相同的对照日上，K 线与逐笔聚合也有数十到数百分钟不一致（开盘价约定
    不同，以及成交被归入相邻分钟、全天合计相等），即逐笔聚合本身不能逐分钟复原 K 线，v1 区分不了“归档缺陷”与“背景差异”。
    因此改用规则 v2（只涉及数据来源，不涉及任何策略结果）：日度 1m 整日替换月度 1m，当且仅当
      (a) 日度 1m 的 24 根 1H 聚合与仓库 1H 的 OHLC 逐位一致；
      (b) 逐笔归档对日度 1m 的每个有成交分钟都有记录（无空洞），且两者全天成交量相对差 ≤ 1e-9；
          逐笔归档有空洞时 (b) 判为不成立（保守：不替换）；
      (c) 日度 1m 没有“记为零成交、逐笔归档却有成交”的分钟；
    且月度 1m 违反 (a) 或 (c)。两条规则的判定都写入记录；对照日的背景差异与其归档来源一并记录。
    """
    daily, dsrc = fetch_daily_1m(day)
    agg, asrc = fetch_aggtrades_1m(day)
    traded = lambda d: d[d["volume"] > 0]          # 无成交的分钟在逐笔归档中没有记录
    silent_m = _silent_traded_minutes(monthly_day, agg)
    silent_d = _silent_traded_minutes(daily, agg)
    holes_d = _agg_holes(daily, agg)
    rec = {"day": str(day.date()), "daily_1m_archive": dsrc, "aggtrades_archive": asrc,
           "zero_volume_minutes": {"monthly": int((monthly_day["volume"] == 0).sum()),
                                   "daily": int((daily["volume"] == 0).sum())},
           "silent_traded_minutes": {"monthly": int(len(silent_m)), "daily": int(len(silent_d)),
                                     "monthly_range": [str(silent_m.min()), str(silent_m.max())] if len(silent_m) else None,
                                     "aggtrades_volume_in_monthly_silent_minutes":
                                         float(agg.loc[silent_m, "volume"].sum()) if len(silent_m) else 0.0},
           "aggtrades_holes_vs_daily": {"minutes": int(len(holes_d)), "ranges": _ranges(holes_d)},
           "monthly_vs_daily": _minute_diff(monthly_day, daily),
           "monthly_vs_aggtrades": _minute_diff(traded(monthly_day), agg),
           "daily_vs_aggtrades": _minute_diff(traded(daily), agg),
           "day_volume": {"monthly": float(monthly_day["volume"].sum()), "daily": float(daily["volume"].sum()),
                          "aggtrades": float(agg["volume"].sum()), "repo_1h": float(h1_day["volume"].sum())},
           "hours_mismatch_monthly_vs_1h": [str(t) for t in ohlc_mismatch_index(agg_ohlcv(monthly_day, "1h"), h1_day)],
           "hours_mismatch_daily_vs_1h": [str(t) for t in ohlc_mismatch_index(agg_ohlcv(daily, "1h"), h1_day)],
           "hours_mismatch_aggtrades_vs_1h": [str(t) for t in ohlc_mismatch_index(agg_ohlcv(agg, "1h"), h1_day)]}
    dv = rec["daily_vs_aggtrades"]
    rec["rule_v1_strict_pass"] = bool(len(daily) == 1440 and dv["ohlc_mismatch_minutes"] == 0
                                      and dv["volume_mismatch_minutes_abs_gt_1e-6"] == 0 and dv["only_in_first"] == 0
                                      and not rec["hours_mismatch_daily_vs_1h"])
    vol_ok = (not len(holes_d)) and \
        abs(rec["day_volume"]["daily"] - rec["day_volume"]["aggtrades"]) <= 1e-9 * rec["day_volume"]["aggtrades"]
    daily_ok = len(daily) == 1440 and not rec["hours_mismatch_daily_vs_1h"] and vol_ok and not len(silent_d)
    monthly_bad = bool(rec["hours_mismatch_monthly_vs_1h"]) or bool(len(silent_m))
    rec["rule_v2"] = {"a_daily_1h_ohlc_equal": not rec["hours_mismatch_daily_vs_1h"],
                      "b_aggtrades_complete_and_day_volume_equal": vol_ok,
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


def _write(df: pd.DataFrame, key: str, cache_dir: Path = CACHE_DIR) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / CACHE_FILES[key]
    df.to_parquet(p, compression="zstd")
    rel = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    return {"path": rel, "rows": int(len(df)), "first_open_time": str(df.index.min()),
            "last_open_time": str(df.index.max()), "content_sha256": content_sha256(df),
            "parquet_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def _manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} missing; build with `python -m impulse_study.data_1m --build`")
    return json.loads(MANIFEST_PATH.read_text())


def _load(key: str, verify: bool) -> pd.DataFrame:
    p = CACHE_DIR / CACHE_FILES[key]
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; rebuild with `python -m impulse_study.data_1m --rebuild` "
                                "(official archives pinned by SHA256 in data/manifest_impulse_1m.json)")
    df = pd.read_parquet(p)
    df.index = pd.DatetimeIndex(df.index).as_unit("ns")
    if verify:
        want = _manifest()["caches"][key]["content_sha256"]
        got = content_sha256(df)
        if got != want:
            raise ValueError(f"{p}: content hash {got} differs from manifest {want}")
    return df


def load_perp_1m(verify: bool = True) -> pd.DataFrame:
    """IS 永续 1m（月度归档 + 经裁决的日度替换；列 src：0 = 月度，1 = 日度替换）。"""
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


def load_repo_1h_is() -> pd.DataFrame:
    """仓库永续 1H（data/processed，覆盖 2020–2026）：以谓词 open_time < IS_END 读取，IS 之后的行不返回给程序；
    不读取资金费文件。"""
    df = pd.read_parquet(REPO_1H_PATH, filters=[("open_time", "<", IS_END)])
    df.index = pd.DatetimeIndex(df.index).as_unit("ns")
    assert_within_is(df.index, "repo 1H")
    return df


def load_repo_spot_1h_warmup(start: str = "2019-09-01") -> pd.DataFrame:
    """仓库现货 1H（覆盖 2019–2026）：以谓词 [start, IS_START) 读取，只供预热数据的一致性核对。"""
    df = pd.read_parquet(REPO_SPOT_1H_PATH, filters=[("open_time", ">=", pd.Timestamp(start, tz="UTC")),
                                                     ("open_time", "<", IS_START)])
    df.index = pd.DatetimeIndex(df.index).as_unit("ns")
    assert_before_is(df.index, "repo spot 1H warm-up")
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
# 构建（首次）与复现（按 manifest）
# ----------------------------------------------------------------------------
def _assemble(frames: dict) -> dict:
    """把下载得到的原始帧整理为各缓存（IS / 预热切片），与构建方式无关。"""
    perp = frames["perp_1m"]
    if perp.index.duplicated().any():
        raise ValueError("duplicate 1m open times in the monthly archives")
    perp = perp[(perp.index >= IS_START) & (perp.index < IS_END)].copy()
    perp["src"] = np.int8(0)
    p15 = frames["perp_15m_official"]
    p1d = frames["perp_1d_official"]
    s1d, s1m = frames["spot_1d_warmup"], frames["spot_1m_warmup"]
    out = {"perp_1m": perp, "perp_15m_official": p15[(p15.index >= IS_START) & (p15.index < IS_END)],
           "perp_1d_official": p1d[(p1d.index >= IS_START) & (p1d.index < IS_END)],
           "spot_1d_warmup": s1d[s1d.index < IS_START], "spot_1m_warmup": s1m[s1m.index < IS_START]}
    assert_before_is(out["spot_1d_warmup"].index, "spot 1d warm-up")
    assert_before_is(out["spot_1m_warmup"].index, "spot 1m warm-up")
    return out


def _replace_day(perp: pd.DataFrame, daily: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    daily = daily.copy()
    daily["src"] = np.int8(1)
    keep = (perp.index < day) | (perp.index >= day + pd.Timedelta(days=1))
    return pd.concat([perp[keep], daily[perp.columns]]).sort_index(kind="stable")


FAMILIES = {"perp_1m": ("futures/um", "1m", IS_MONTHS), "perp_15m_official": ("futures/um", "15m", IS_MONTHS),
            "perp_1d_official": ("futures/um", "1d", IS_MONTHS), "spot_1d_warmup": ("spot", "1d", SPOT_1D_WARMUP_MONTHS),
            "spot_1m_warmup": ("spot", "1m", SPOT_1M_WARMUP_MONTHS)}


def build(workers: int = 8, force_new_manifest: bool = False) -> dict:
    """首次构建：在线 CHECKSUM 校验 → 裁决 → 写缓存与 manifest。manifest 已存在时须显式允许覆盖。"""
    old = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else None
    if old is not None and not force_new_manifest:
        raise SystemExit(f"{MANIFEST_PATH} exists; use --rebuild to reproduce it, or --build --force-new-manifest to "
                         "re-sign a fresh download (changes are reported)")
    df1h = load_repo_1h_is()
    frames, files = {}, {}
    for key, (market, interval, months) in FAMILIES.items():
        print(f"[{key}] {market} {interval} {months[0]} .. {months[1]}", flush=True)
        frames[key], files[key] = fetch_monthly(market, interval, months, workers)
    raw_stats = {"rows_downloaded": int(len(frames["perp_1m"])),
                 "duplicate_open_times": int(frames["perp_1m"].index.duplicated().sum()),
                 "rows_outside_is": int(((frames["perp_1m"].index < IS_START) | (frames["perp_1m"].index >= IS_END)).sum())}
    data = _assemble(frames)
    perp = data["perp_1m"]
    bad_hours = ohlc_mismatch_index(agg_ohlcv(perp, "1h"), df1h)
    adjud = []
    for day in sorted({t.floor("D") for t in bad_hours}):
        print(f"[perp 1m] adjudicating {day.date()} (monthly 1m vs 1H mismatch)", flush=True)
        mday = perp[(perp.index >= day) & (perp.index < day + pd.Timedelta(days=1))]
        hday = df1h[(df1h.index >= day) & (df1h.index < day + pd.Timedelta(days=1))]
        rec = adjudicate_day(day, mday, hday, monthly_all=perp)
        daily = rec.pop("_daily_df")
        if rec["verdict"] == "replace_with_daily_archive":
            perp = _replace_day(perp, daily, day)
        adjud.append(rec)
    data["perp_1m"] = perp
    still_bad = ohlc_mismatch_index(agg_ohlcv(perp, "1h"), df1h)
    assert_within_is(perp.index, "perp 1m")
    caches = {k: _write(v, k) for k, v in data.items()}
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
        "provider": "binance_vision (official archives, SHA256 checked against the published .CHECKSUM at download)",
        "raw_download_stats_perp_1m": raw_stats,
        "files": files,
        "adjudication_rule": "v2 (v1 recorded per day as rule_v1_strict_pass; see impulse_study.data_1m.adjudicate_day): "
                             "a calendar day whose monthly-1m hourly aggregate differs from the repository 1H OHLC is "
                             "replaced by the official daily 1m archive iff (a) the daily archive's 24 hourly aggregates "
                             "equal the repository 1H OHLC exactly, (b) the aggTrades archive covers every traded minute "
                             "of the daily archive and the day volumes agree (rel. <= 1e-9), (c) the daily archive has no "
                             "zero-volume minute in which aggTrades records trades, and the monthly archive violates (a) "
                             "or (c). v1 required minute-exact equality with aggTrades, which fails even on control days "
                             "where the monthly and daily archives are bit-identical",
        "adjudicated_days": adjud,
        "hours_ohlc_mismatch_vs_1h_after_adjudication": [str(t) for t in still_bad],
        "caches": caches,
        "column_src": "perp_1m.src: 0 = monthly archive, 1 = official daily archive (adjudicated replacement)",
        "rebuild": "python -m impulse_study.data_1m --rebuild [--offline]  (pinned to the SHA256 values above; raw zips "
                   "under data/raw/ and parquet under data/processed_1m/ are git-ignored)",
    }
    if old is not None:
        changed = {k: (old.get("caches", {}).get(k, {}).get("content_sha256"), v["content_sha256"])
                   for k, v in caches.items() if old.get("caches", {}).get(k, {}).get("content_sha256") != v["content_sha256"]}
        man["content_hash_changes_vs_previous_manifest"] = changed
        print("content hash changes vs previous manifest:", changed or "none", flush=True)
    MANIFEST_PATH.write_text(json.dumps(man, indent=1, ensure_ascii=False))
    return man


def rebuild_from_manifest(cache_dir: Path = CACHE_DIR, workers: int = 8) -> dict:
    """按 manifest 复现：每个文件的 SHA256 必须等于记录值；内容哈希全部相等后才写缓存；不改写 manifest。"""
    man = _manifest()
    frames = {}
    for key in FAMILIES:
        recs = man["files"][key]
        rels = [r["url"].removeprefix(VISION_BASE + "/") for r in recs]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda a: fetch_archive(*a), zip(rels, [r["sha256"] for r in recs])))
        frames[key] = pd.concat([_parse(b) for b, _ in results]).sort_index(kind="stable")
    data = _assemble(frames)
    perp = data["perp_1m"]
    for rec in man["adjudicated_days"]:
        if rec["verdict"] != "replace_with_daily_archive":
            continue
        day = pd.Timestamp(rec["day"], tz="UTC")
        daily, _ = fetch_daily_1m(day, pinned_sha256=rec["daily_1m_archive"]["sha256"])
        perp = _replace_day(perp, daily, day)
    data["perp_1m"] = perp
    got = {k: content_sha256(v) for k, v in data.items()}
    bad = {k: (man["caches"][k]["content_sha256"], g) for k, g in got.items() if g != man["caches"][k]["content_sha256"]}
    if bad:
        raise ValueError(f"rebuilt content differs from the manifest: {bad}")
    for k, v in data.items():
        _write(v, k, cache_dir)
    return {k: True for k in got}


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
    ap = argparse.ArgumentParser(description="Build / rebuild / verify the impulse-study 1m cache (BTC IS only).")
    ap.add_argument("--build", action="store_true", help="fresh download verified against the live CHECKSUM files")
    ap.add_argument("--force-new-manifest", action="store_true", help="allow --build to overwrite an existing manifest")
    ap.add_argument("--rebuild", action="store_true", help="reproduce the caches from the SHA256-pinned manifest")
    ap.add_argument("--offline", action="store_true", help="with --rebuild: never touch the network")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.offline:
        OFFLINE = True
    if a.build:
        m = build(force_new_manifest=a.force_new_manifest)
        print(json.dumps({k: v for k, v in m.items() if k not in ("files", "adjudicated_days")}, indent=1,
                         ensure_ascii=False)[:4000])
    if a.rebuild:
        print(json.dumps(rebuild_from_manifest(), indent=1))
    if a.verify:
        print(json.dumps(verify(), indent=1))
    if not (a.build or a.rebuild or a.verify):
        raise SystemExit("nothing to do: pass --build, --rebuild or --verify")

#!/usr/bin/env python3
"""
data_loader.py — 数据工程模块（Phase 0）

数据源优先级（自动回退）
------------------------
1. 本地 Parquet 缓存 ``data/processed/*.parquet``（来源与校验和记录在 ``data/manifest.json``）
2. Binance 官方公开数据归档 ``data.binance.vision``（USDⓈ-M 永续 / 现货月度 zip，逐文件 SHA256 校验）
3. Binance REST API（``fapi.binance.com`` / ``api.binance.com`` 的 ``/klines`` 分页拉取）
4. OKX 公开 REST API（``/api/v5/market/history-candles`` 分页拉取，USDT 本位永续）

本研究的主数据为 Binance USDⓈ-M 永续合约（与 taker 5bp / maker 2bp 的费率结构、≤1.5x 杠杆一致），
资金费率（funding）同样取自官方归档，供回测引擎计入持仓成本。现货 1H 数据仅用于
(a) 慢速价格类指标的 2019 年预热，(b) 永续-现货基差数据质量交叉校验。

时间戳约定（全项目统一）
------------------------
* 所有 K 线 DataFrame 的索引均为 **UTC 开盘时间 open_time**（tz-aware）。
* ``close_time = open_time + interval``，即该 K 线完整收盘、数据可被使用的最早时刻。
  （Binance 原始 close_time 为 open_time + interval - 1ms，此处统一取整到收盘瞬间。）
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # 原始 zip（不入库，可按 manifest 重新下载复现）
PROC_DIR = DATA_DIR / "processed"     # 处理后的 Parquet 缓存（入库，便于审查复现）
MANIFEST_PATH = DATA_DIR / "manifest.json"

VISION_BASE = "https://data.binance.vision/data"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
OKX_INST = {"BTCUSDT": "BTC-USDT-SWAP", "ETHUSDT": "ETH-USDT-SWAP"}

# 研究区间：[2020-01-01, 2026-11-01) UTC；永续归档自 2020-01 起
PERP_START_YM, END_YM = "2020-01", "2026-10"
SPOT_START_YM = "2019-01"             # 现货额外下载 2019 全年用于慢速价格指标预热
RESEARCH_START = pd.Timestamp("2020-01-01", tz="UTC")
RESEARCH_END = pd.Timestamp("2026-11-01", tz="UTC")   # 右开区间

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
KEEP_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]
INTERVAL_TD = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "ppvp-quant-research/1.0"})


# ----------------------------------------------------------------------------
# 通用下载工具
# ----------------------------------------------------------------------------
def month_range(start_ym: str, end_ym: str) -> list[str]:
    """闭区间月份序列，例如 ('2020-01', '2020-03') -> ['2020-01', '2020-02', '2020-03']。"""
    return [p.strftime("%Y-%m") for p in pd.period_range(start_ym, end_ym, freq="M")]


def _http_get(url: str, params: dict | None = None, retries: int = 6, timeout: int = 60) -> requests.Response:
    """带指数退避的 GET；对 429/5xx/网络错误重试，对 4xx（含 451 地区限制）立即抛出。"""
    delay = 2.0
    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            r = _SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                last_exc = RuntimeError(f"HTTP {r.status_code} for {url}")
            else:
                raise requests.HTTPError(f"HTTP {r.status_code} for {url}", response=r)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
        time.sleep(delay)
        delay = min(delay * 2, 30)
    raise RuntimeError(f"GET failed after {retries} retries: {url}: {last_exc}")


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _vision_path(market: str, kind: str, symbol: str, ym: str, interval: str | None = None) -> str:
    """market ∈ {'futures/um', 'spot'}; kind ∈ {'klines', 'fundingRate'}。"""
    if kind == "klines":
        return f"{market}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
    if kind == "fundingRate":
        return f"{market}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"
    raise ValueError(kind)


def fetch_vision_zip(rel_path: str) -> tuple[bytes, dict]:
    """下载（或读取本地缓存）一个归档 zip，并用官方 .CHECKSUM 做 SHA256 校验。返回 (CSV 字节, 来源记录)。"""
    url = f"{VISION_BASE}/{rel_path}"
    local = RAW_DIR / rel_path
    expected = _http_get(url + ".CHECKSUM").text.split()[0].strip().lower()
    if local.exists() and _sha256(local.read_bytes()) == expected:
        blob = local.read_bytes()
    else:
        blob = _http_get(url).content
        got = _sha256(blob)
        if got != expected:
            raise ValueError(f"SHA256 mismatch for {url}: expected {expected}, got {got}")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
        if len(names) != 1:
            raise ValueError(f"unexpected zip content {names} in {url}")
        csv_bytes = z.read(names[0])
    return csv_bytes, {"url": url, "sha256": expected, "checksum_verified": True}


# ----------------------------------------------------------------------------
# CSV 解析
# ----------------------------------------------------------------------------
def _has_header(csv_bytes: bytes) -> bool:
    first_token = csv_bytes.split(b"\n", 1)[0].split(b",")[0].strip()
    return not first_token.isdigit()


def parse_kline_csv(csv_bytes: bytes) -> pd.DataFrame:
    """解析 Binance kline CSV（兼容有/无表头、毫秒/微秒时间戳），返回以 UTC open_time 为索引的 DataFrame。"""
    df = pd.read_csv(io.BytesIO(csv_bytes), header=0 if _has_header(csv_bytes) else None)
    if df.shape[1] != len(KLINE_COLS):
        raise ValueError(f"unexpected kline column count {df.shape[1]}")
    df.columns = KLINE_COLS
    ot = df["open_time"].astype("int64").to_numpy()
    ct = df["close_time"].astype("int64").to_numpy()
    if ot.max() > 10**14:          # 现货归档自 2025-01 起改为微秒
        ot, ct = ot // 1000, ct // 1000
    out = df[KEEP_COLS].astype("float64")
    out.index = pd.to_datetime(ot, unit="ms", utc=True)
    out.index.name = "open_time"
    out["raw_close_time_ms"] = ct
    return out


def parse_funding_csv(csv_bytes: bytes) -> pd.DataFrame:
    """解析资金费率 CSV；calc_time 存在毫秒级抖动（如 ...800002），四舍五入到整分钟。"""
    df = pd.read_csv(io.BytesIO(csv_bytes), header=0 if _has_header(csv_bytes) else None)
    df.columns = ["calc_time", "funding_interval_hours", "funding_rate"][: df.shape[1]]
    t = pd.to_datetime(df["calc_time"].astype("int64"), unit="ms", utc=True).dt.round("min")
    out = pd.DataFrame(
        {"funding_rate": df["funding_rate"].astype("float64").to_numpy(),
         "funding_interval_hours": df["funding_interval_hours"].astype("int64").to_numpy()},
        index=pd.DatetimeIndex(t, name="funding_time"),
    )
    return out


# ----------------------------------------------------------------------------
# 数据源 2：Binance Vision 归档
# ----------------------------------------------------------------------------
def fetch_vision_klines(symbol: str, interval: str, market: str, start_ym: str, end_ym: str,
                        workers: int = 8) -> tuple[pd.DataFrame, list[dict]]:
    paths = [_vision_path(market, "klines", symbol, ym, interval) for ym in month_range(start_ym, end_ym)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(fetch_vision_zip, paths))
    frames, sources = [], []
    for csv_bytes, src in results:
        f = parse_kline_csv(csv_bytes)
        src["rows"] = int(len(f))
        frames.append(f)
        sources.append(src)
    df = pd.concat(frames).sort_index()
    return df, sources


def fetch_vision_funding(symbol: str, start_ym: str, end_ym: str, workers: int = 8) -> tuple[pd.DataFrame, list[dict]]:
    paths = [_vision_path("futures/um", "fundingRate", symbol, ym) for ym in month_range(start_ym, end_ym)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(fetch_vision_zip, paths))
    frames, sources = [], []
    for csv_bytes, src in results:
        f = parse_funding_csv(csv_bytes)
        src["rows"] = int(len(f))
        frames.append(f)
        sources.append(src)
    return pd.concat(frames).sort_index(), sources


# ----------------------------------------------------------------------------
# 数据源 3/4：REST 回退（Binance fapi -> OKX）
# ----------------------------------------------------------------------------
def fetch_binance_rest_klines(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Binance USDⓈ-M REST 分页拉取（本环境返回 451 地区限制时由上层自动回退）。"""
    step = INTERVAL_TD[interval]
    rows, cursor = [], int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor < end_ms:
        r = _http_get("https://fapi.binance.com/fapi/v1/klines",
                      params={"symbol": symbol, "interval": interval, "startTime": cursor,
                              "endTime": end_ms - 1, "limit": 1500})
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + int(step.total_seconds() * 1000)
    if not rows:
        return pd.DataFrame(columns=KEEP_COLS)
    csv = "\n".join(",".join(str(x) for x in row) for row in rows).encode()
    return parse_kline_csv(csv)


def fetch_okx_klines(symbol: str, start: pd.Timestamp, end: pd.Timestamp, bar: str = "1H") -> pd.DataFrame:
    """OKX USDT 永续 1H K 线（history-candles 每次 100 根，按 after 游标向过去翻页）。

    OKX 字段: [ts, o, h, l, c, vol(张), volCcy(币), volCcyQuote(USDT), confirm]；volume 取 volCcy（币本位数量）。
    """
    inst = OKX_INST[symbol]
    rows: list[list[str]] = []
    after = int(end.timestamp() * 1000)
    start_ms = int(start.timestamp() * 1000)
    while after > start_ms:
        r = _http_get("https://www.okx.com/api/v5/market/history-candles",
                      params={"instId": inst, "bar": bar, "after": str(after), "limit": "100"})
        data = r.json().get("data", [])
        if not data:
            break
        rows.extend(data)
        after = int(data[-1][0])
        time.sleep(0.12)  # 公共接口限频 20 次/2 秒
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
    df = df[df["confirm"] == "1"]
    out = pd.DataFrame({
        "open": df["open"].astype(float).to_numpy(), "high": df["high"].astype(float).to_numpy(),
        "low": df["low"].astype(float).to_numpy(), "close": df["close"].astype(float).to_numpy(),
        "volume": df["volCcy"].astype(float).to_numpy(), "quote_volume": df["volCcyQuote"].astype(float).to_numpy(),
    }, index=pd.to_datetime(df["ts"].astype("int64").to_numpy(), unit="ms", utc=True))
    out.index.name = "open_time"
    out = out[~out.index.duplicated()].sort_index()
    return out[(out.index >= start) & (out.index < end)]


def fetch_perp_klines_with_fallback(symbol: str, interval: str, start_ym: str, end_ym: str) -> tuple[pd.DataFrame, dict]:
    """按 Vision -> Binance REST -> OKX 的顺序获取永续 K 线，返回 (df, 来源说明)。"""
    errors = []
    try:
        df, sources = fetch_vision_klines(symbol, interval, "futures/um", start_ym, end_ym)
        return df, {"provider": "binance_vision_futures_um", "files": sources}
    except Exception as exc:  # noqa: BLE001
        errors.append(f"binance_vision: {exc}")
    start = pd.Timestamp(f"{start_ym}-01", tz="UTC")
    end = (pd.Period(end_ym, freq="M") + 1).to_timestamp().tz_localize("UTC")
    try:
        df = fetch_binance_rest_klines(symbol, interval, start, end)
        return df, {"provider": "binance_rest_fapi", "errors": errors}
    except Exception as exc:  # noqa: BLE001
        errors.append(f"binance_rest: {exc}")
    if interval != "1h":
        raise RuntimeError(f"no fallback source for interval {interval}: {errors}")
    df = fetch_okx_klines(symbol, start, end)
    return df, {"provider": "okx_rest_swap", "errors": errors}


# ----------------------------------------------------------------------------
# 重采样与数据质量检查
# ----------------------------------------------------------------------------
def add_close_time(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    df = df.copy()
    df["close_time"] = df.index + INTERVAL_TD[interval]
    return df


def resample_ohlcv(df_1h: pd.DataFrame, rule: str = "4h") -> pd.DataFrame:
    """把 1H K 线聚合为 UTC 对齐（00/04/08/12/16/20 点开盘）的高周期 K 线。

    label=left/closed=left：索引为开盘时间，只聚合 [open_time, open_time+rule) 内的 1H K 线；
    ``n_bars`` 记录实际参与聚合的 1H 根数（交易所停机缺数时 < 4）。
    """
    g = df_1h.resample(rule, origin="epoch", label="left", closed="left")
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
        "quote_volume": g["quote_volume"].sum(), "n_bars": g["close"].count(),
    })
    out = out[out["n_bars"] > 0].copy()
    out["close_time"] = out.index + pd.Timedelta(rule)
    return out


def _ranges(ts: pd.DatetimeIndex, step: pd.Timedelta) -> list[list[str]]:
    """把离散缺失时间戳合并为连续区间 [[first, last, count], ...]。"""
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
    """与 pandas 时间分辨率（ns/us/ms）无关的 epoch 毫秒。"""
    return ((idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy(dtype="int64")


# ----------------------------------------------------------------------------
# 数据清洗：维护期占位 K 线、归档缺口回退补齐
# ----------------------------------------------------------------------------
ARCHIVE_GAP_MIN_HOURS = 6   # ≥6 小时的连续缺口视为“归档发布缺口”（交易实际发生），用回退源补齐；
                            # 更短的缺口视为真实的交易所维护停机，保持缺失，不伪造 K 线。


def drop_placeholder_bars(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """剔除维护期间的零成交占位 K 线（volume==0 且 O=H=L=C）。

    依据：Binance 官方 4H 聚合同样排除了这类 K 线（2024-10-28 20:00 的 4H 开盘价取自 21:00 的 1H）。
    """
    mask = (df["volume"] == 0) & (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
    return df.loc[~mask].copy(), [str(t) for t in df.index[mask]]


def fill_archive_gaps(df: pd.DataFrame, symbol: str, min_hours: int = ARCHIVE_GAP_MIN_HOURS) -> tuple[pd.DataFrame, list[dict]]:
    """用 OKX USDT 永续 1H 补齐 Binance 归档中 ≥min_hours 的连续缺口，并逐行标记 fill_source。

    成交量口径对齐：缩放系数 = 缺口开始前 168 根 1H 上 (Binance 成交量 / OKX 成交量) 的中位数，
    只使用缺口之前的数据，避免数据准备环节引入任何未来信息。
    """
    step = INTERVAL_TD["1h"]
    df = df.copy()
    df["fill_source"] = "binance_um"
    grid = pd.date_range(RESEARCH_START, RESEARCH_END - step, freq=step, tz="UTC")
    records = []
    for s, e, n in _ranges(grid.difference(df.index), step):
        if n < min_hours:
            continue
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        okx_gap = fetch_okx_klines(symbol, s, e + step)
        okx_hist = fetch_okx_klines(symbol, s - 168 * step, s)
        bin_hist = df.loc[(df.index >= s - 168 * step) & (df.index < s), "volume"]
        common = bin_hist.index.intersection(okx_hist.index)
        ratio = float(np.median(bin_hist.loc[common].to_numpy() / okx_hist.loc[common, "volume"].to_numpy()))
        fill = okx_gap.copy()
        fill["volume"] *= ratio
        fill["quote_volume"] *= ratio
        fill["count"] = np.nan
        fill["taker_buy_volume"] = np.nan
        fill["fill_source"] = "okx_swap"
        if "raw_close_time_ms" in df.columns:
            fill["raw_close_time_ms"] = _epoch_ms(fill.index) + int(step.total_seconds() * 1000) - 1
        df = pd.concat([df, fill[df.columns]]).sort_index()
        records.append({"gap_start": str(s), "gap_end": str(e), "missing_hours": n, "filled_hours": int(len(fill)),
                        "source": "okx_rest " + OKX_INST[symbol], "volume_scale_ratio_binance_over_okx": ratio,
                        "scale_ratio_window_hours": int(len(common))})
    return df, records


def qa_fill_vs_spot(perp: pd.DataFrame, spot: pd.DataFrame) -> dict:
    """补齐行与 Binance 现货（独立来源）同时段收盘价差异，用于评估补数质量。"""
    f = perp[perp["fill_source"] != "binance_um"]
    if f.empty:
        return {}
    common = f.index.intersection(spot.index)
    d = (f.loc[common, "close"] / spot.loc[common, "close"] - 1) * 1e4
    return {"filled_rows": int(len(f)), "compared_rows": int(len(common)),
            "close_diff_vs_binance_spot_bp_median": float(d.median()), "close_diff_bp_max_abs": float(d.abs().max())}


def qa_ohlcv(df: pd.DataFrame, interval: str, start: pd.Timestamp, end_excl: pd.Timestamp) -> dict:
    """K 线质量体检：重复/乱序/缺口/OHLC 逻辑/非正价格/成交量/收盘时间一致性/极端收益。"""
    step = INTERVAL_TD[interval]
    idx = df.index
    grid = pd.date_range(start, end_excl - step, freq=step, tz="UTC")
    missing = grid.difference(idx)
    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)
    bad_ohlc = (df["high"] < body_hi) | (df["low"] > body_lo) | (df["high"] < df["low"])
    lr = np.log(df["close"]).diff()
    res = {
        "rows": int(len(df)),
        "first_open_time": str(idx.min()),
        "last_open_time": str(idx.max()),
        "expected_rows": int(len(grid)),
        "duplicates": int(idx.duplicated().sum()),
        "monotonic_increasing": bool(idx.is_monotonic_increasing),
        "missing_bars": int(len(missing)),
        "missing_ranges": _ranges(missing, step),
        "off_grid_bars": int(len(idx.difference(grid))),
        "bad_ohlc_rows": int(bad_ohlc.sum()),
        "non_positive_price_rows": int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
        "negative_volume_rows": int((df["volume"] < 0).sum()),
        "zero_volume_rows": int((df["volume"] == 0).sum()),
        "max_abs_log_return": float(lr.abs().max()),
        "max_abs_log_return_at": str(lr.abs().idxmax()),
    }
    if "raw_close_time_ms" in df.columns:
        expect_ct = _epoch_ms(idx) + int(step.total_seconds() * 1000) - 1
        res["close_time_mismatch_rows"] = int((df["raw_close_time_ms"].to_numpy() != expect_ct).sum())
    return res


def qa_resample_vs_official(res_4h: pd.DataFrame, off_4h: pd.DataFrame) -> dict:
    """1H 自行聚合的 4H 与 Binance 官方 4H 逐根对比（价格应逐位一致，成交量仅浮点求和误差）。"""
    common = res_4h.index.intersection(off_4h.index)
    a, b = res_4h.loc[common], off_4h.loc[common]
    out = {"common_bars": int(len(common)),
           "only_in_resampled": int(len(res_4h.index.difference(off_4h.index))),
           "only_in_official": int(len(off_4h.index.difference(res_4h.index)))}
    for c in ["open", "high", "low", "close"]:
        out[f"{c}_mismatch_bars"] = int((a[c] != b[c]).sum())
    rel = (a["volume"] - b["volume"]).abs() / b["volume"].where(b["volume"] > 0)
    out["volume_max_rel_diff"] = float(rel.max())
    out["volume_mismatch_bars_rel_gt_1e-6"] = int((rel > 1e-6).sum())
    out["resampled_bars_with_lt4_1h"] = int((res_4h["n_bars"] < 4).sum())
    return out


def qa_funding(f: pd.DataFrame) -> dict:
    gaps = f.index.to_series().diff().dropna()
    return {
        "rows": int(len(f)), "first": str(f.index.min()), "last": str(f.index.max()),
        "duplicates": int(f.index.duplicated().sum()),
        "interval_hours_counts": {str(k): int(v) for k, v in f["funding_interval_hours"].value_counts().items()},
        "max_gap_hours": float(gaps.max() / pd.Timedelta(hours=1)),
        "mean_rate": float(f["funding_rate"].mean()),
        "min_rate": float(f["funding_rate"].min()), "max_rate": float(f["funding_rate"].max()),
        "annualized_mean_long_cost_pct": float(f["funding_rate"].mean() * (365 * 24 / f["funding_interval_hours"].mean()) * 100),
    }


def qa_basis(perp: pd.DataFrame, spot: pd.DataFrame) -> dict:
    common = perp.index.intersection(spot.index)
    basis_bp = (perp.loc[common, "close"] / spot.loc[common, "close"] - 1) * 1e4
    r1 = np.log(perp.loc[common, "close"]).diff()
    r2 = np.log(spot.loc[common, "close"]).diff()
    return {
        "common_bars": int(len(common)),
        "basis_bp_median": float(basis_bp.median()),
        "basis_bp_p01": float(basis_bp.quantile(0.01)), "basis_bp_p99": float(basis_bp.quantile(0.99)),
        "basis_bp_max_abs": float(basis_bp.abs().max()),
        "hourly_log_return_corr": float(r1.corr(r2)),
    }


def qa_okx_crosscheck(perp: pd.DataFrame, okx: pd.DataFrame) -> dict:
    common = perp.index.intersection(okx.index)
    d = (perp.loc[common, "close"] / okx.loc[common, "close"] - 1) * 1e4
    return {"window": [str(okx.index.min()), str(okx.index.max())], "common_bars": int(len(common)),
            "close_diff_bp_median_abs": float(d.abs().median()), "close_diff_bp_max_abs": float(d.abs().max()),
            "hourly_log_return_corr": float(np.log(perp.loc[common, "close"]).diff().corr(
                np.log(okx.loc[common, "close"]).diff()))}


# ----------------------------------------------------------------------------
# 缓存构建与读取
# ----------------------------------------------------------------------------
def _proc_path(name: str) -> Path:
    return PROC_DIR / f"{name}.parquet"


def _write_parquet(df: pd.DataFrame, name: str) -> dict:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    p = _proc_path(name)
    df.to_parquet(p, compression="zstd")
    return {"path": str(p.relative_to(ROOT)), "rows": int(len(df)), "sha256": _sha256(p.read_bytes())}


def adjudicate_4h_mismatches(symbol: str, r4h: pd.DataFrame, off4h: pd.DataFrame, vol_tol: float = 1e-3) -> list[dict]:
    """对自聚合 4H 与官方 4H 的差异 K 线，下载当日官方 1m 归档重新聚合，裁决哪一方正确。"""
    common = r4h.index.intersection(off4h.index)
    a, b = r4h.loc[common], off4h.loc[common]
    price_bad = (a[["open", "high", "low", "close"]] != b[["open", "high", "low", "close"]]).any(axis=1)
    vol_bad = (a["volume"] - b["volume"]).abs() / b["volume"] > vol_tol
    out = []
    for t in a.index[price_bad | vol_bad]:
        rec = {"bar_open_time": str(t),
               "resampled_1h": {k: float(a.loc[t, k]) for k in ["open", "high", "low", "close", "volume"]},
               "official_4h": {k: float(b.loc[t, k]) for k in ["open", "high", "low", "close", "volume"]}}
        try:
            csv_bytes, _ = fetch_vision_zip(
                f"futures/um/daily/klines/{symbol}/1m/{symbol}-1m-{t.strftime('%Y-%m-%d')}.zip")
            m = parse_kline_csv(csv_bytes)
            m = m[(m.index >= t) & (m.index < t + pd.Timedelta(hours=4)) & (m["volume"] > 0)]
            agg = {"open": float(m["open"].iloc[0]), "high": float(m["high"].max()), "low": float(m["low"].min()),
                   "close": float(m["close"].iloc[-1]), "volume": float(m["volume"].sum())}
            rec["official_1m_aggregate"] = agg
            close_to = lambda x: all(abs(x[k] - agg[k]) <= 1e-9 * max(1.0, abs(agg[k])) for k in ["open", "high", "low", "close"]) \
                and abs(x["volume"] - agg["volume"]) / agg["volume"] < vol_tol
            rec["verdict"] = ("resampled_1h_correct" if close_to(rec["resampled_1h"]) else
                              "official_4h_correct" if close_to(rec["official_4h"]) else "neither_exact")
        except Exception as exc:  # noqa: BLE001
            rec["verdict"] = f"unverifiable: {exc}"
        out.append(rec)
    return out


def build_cache(okx_check_window: tuple[str, str] = ("2024-03-01", "2024-03-15")) -> dict:
    """下载全部原始数据 -> 校验 -> 质量体检 -> 清洗/补缺 -> 写 Parquet 缓存与 manifest.json。"""
    manifest: dict = {"generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                      "research_window": [str(RESEARCH_START), str(RESEARCH_END)],
                      "timestamp_convention": "index = UTC open_time; close_time = open_time + interval",
                      "cleaning_policy": {
                          "placeholder_bars": "volume==0 且 O=H=L=C 的维护占位 K 线剔除（与官方 4H 聚合口径一致）",
                          "archive_gaps": f">= {ARCHIVE_GAP_MIN_HOURS}h 的连续缺口用 OKX USDT 永续补齐，成交量按缺口前 168h 中位比值缩放；"
                                          "更短缺口视为真实停机保持缺失",
                          "4h_source": "策略使用 1H 本地聚合的 4H（官方 4H 仅作交叉校验，差异由 1m 归档裁决）"},
                      "datasets": {}, "qa": {}}
    for sym in SYMBOLS:
        print(f"[{sym}] perp 1h ...", flush=True)
        p1h_raw, src1 = fetch_perp_klines_with_fallback(sym, "1h", PERP_START_YM, END_YM)
        print(f"[{sym}] perp 4h (official) ...", flush=True)
        p4h, src4 = fetch_perp_klines_with_fallback(sym, "4h", PERP_START_YM, END_YM)
        print(f"[{sym}] funding ...", flush=True)
        fund, srcf = fetch_vision_funding(sym, PERP_START_YM, END_YM)
        print(f"[{sym}] spot 1h ...", flush=True)
        s1h, srcs = fetch_vision_klines(sym, "1h", "spot", SPOT_START_YM, END_YM)

        print(f"[{sym}] cleaning + gap fill ...", flush=True)
        p1h, placeholders = drop_placeholder_bars(p1h_raw)
        p1h, fills = fill_archive_gaps(p1h, sym)

        qa = {
            "perp_1h_raw_archive": qa_ohlcv(p1h_raw, "1h", RESEARCH_START, RESEARCH_END),
            "perp_1h_cleaned": qa_ohlcv(p1h, "1h", RESEARCH_START, RESEARCH_END),
            "dropped_placeholder_bars": placeholders,
            "archive_gap_fills": fills,
            "fill_quality_vs_binance_spot": qa_fill_vs_spot(p1h, s1h),
            "perp_4h_official": qa_ohlcv(p4h, "4h", RESEARCH_START, RESEARCH_END),
            "spot_1h": qa_ohlcv(s1h, "1h", pd.Timestamp(f"{SPOT_START_YM}-01", tz="UTC"), RESEARCH_END),
            "funding": qa_funding(fund),
        }
        binance_only = p1h[p1h["fill_source"] == "binance_um"]
        r4h = resample_ohlcv(binance_only, "4h")
        qa["resampled_4h_vs_official_4h"] = qa_resample_vs_official(r4h, p4h)
        qa["resampled_4h_vs_official_4h_adjudication_1m"] = adjudicate_4h_mismatches(sym, r4h, p4h)
        qa["perp_vs_spot_basis"] = qa_basis(binance_only, s1h)
        try:
            okx = fetch_okx_klines(sym, pd.Timestamp(okx_check_window[0], tz="UTC"),
                                   pd.Timestamp(okx_check_window[1], tz="UTC"))
            qa["okx_independent_crosscheck"] = qa_okx_crosscheck(binance_only, okx)
        except Exception as exc:  # noqa: BLE001
            qa["okx_independent_crosscheck"] = {"error": str(exc)}
        manifest["qa"][sym] = qa

        drop_raw = ["raw_close_time_ms"]
        manifest["datasets"][f"{sym}_perp_1h"] = {
            **_write_parquet(add_close_time(p1h.drop(columns=drop_raw), "1h"), f"{sym}_perp_1h"),
            "provider": src1["provider"], "files": src1.get("files", []), "gap_fills": fills,
            "dropped_placeholder_bars": placeholders}
        manifest["datasets"][f"{sym}_perp_4h_official"] = {
            **_write_parquet(add_close_time(p4h.drop(columns=drop_raw), "4h"), f"{sym}_perp_4h_official"),
            "provider": src4["provider"], "files": src4.get("files", [])}
        manifest["datasets"][f"{sym}_funding"] = {
            **_write_parquet(fund, f"{sym}_funding"), "provider": "binance_vision_futures_um", "files": srcf}
        manifest["datasets"][f"{sym}_spot_1h"] = {
            **_write_parquet(add_close_time(s1h.drop(columns=drop_raw), "1h"), f"{sym}_spot_1h"),
            "provider": "binance_vision_spot", "files": srcs}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def _ensure_cache(name: str) -> Path:
    p = _proc_path(name)
    if not p.exists():
        build_cache()
    return p


def load_klines(symbol: str = "BTCUSDT", interval: str = "1h", market: str = "perp") -> pd.DataFrame:
    """读取缓存 K 线（索引 = UTC open_time，含 close_time 列），返回 2020-2026 全量数据、不做 OOS 截断。

    研究阶段请通过 ``pipeline.load_market`` / ``pipeline.run`` 读取（带 OOS 锁）；直接调用本函数仅限
    数据 QA、防未来函数闸门与 Phase 4。缓存构建中的全部清洗步骤（占位 K 线剔除、缺口填补的量能比例）
    都是逐根或严格向后看的，任何新增的数据层变换也必须满足这一点，否则截断测试无法发现由此引入的未来信息。
    interval='4h' 时由 1H 本地聚合（与官方 4H 已在 QA 中逐根比对一致），保证两个周期数据同源。
    """
    if interval == "1h":
        return pd.read_parquet(_ensure_cache(f"{symbol}_{market}_1h"))
    if interval == "4h":
        return resample_ohlcv(load_klines(symbol, "1h", market).drop(columns=["close_time"]), "4h")
    raise ValueError(interval)


def load_funding(symbol: str = "BTCUSDT") -> pd.DataFrame:
    return pd.read_parquet(_ensure_cache(f"{symbol}_funding"))


def verify_cache() -> dict:
    """核对 Parquet 缓存与 manifest 中记录的 SHA256 是否一致（审查复现用）。"""
    manifest = json.loads(MANIFEST_PATH.read_text())
    return {name: _sha256((ROOT / meta["path"]).read_bytes()) == meta["sha256"]
            for name, meta in manifest["datasets"].items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build / verify the local Parquet data cache.")
    ap.add_argument("--build", action="store_true", help="download, verify and cache all datasets")
    ap.add_argument("--verify", action="store_true", help="verify cached Parquet files against manifest")
    args = ap.parse_args()
    if args.build:
        m = build_cache()
        print(json.dumps(m["qa"], indent=2, ensure_ascii=False))
    if args.verify:
        print(json.dumps(verify_cache(), indent=2))

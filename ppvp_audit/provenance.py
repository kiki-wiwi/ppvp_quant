"""
ppvp_audit/provenance.py — 数据来源标记（用户第 3.1 节；审计 D-3、D-6、D-10）。IS only。

逐根 1H：venue、market_type、source（fill_source）、is_missing（网格上缺失的小时）、is_imputed（非 Binance 永续来源，
如 OKX 补数）、is_partial（疑似只交易了一小部分时间的小时：成交量低于过去 168 小时中位数的 1%，事先写定的启发式阈值，
只用于披露，不改变任何策略判定）。再聚合到 4H（n_bars、n_imputed、n_partial）与每个箱体的分布窗口，供账本引用。
另给出每个机会的市场状态特征是否用到 2020 年之前的现货预热（SMA200 窗口 200 天，365 日特征 365 天）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import IS_END, IS_START

PARTIAL_VOLUME_FRACTION = 0.01


def bar_flags(df1h: pd.DataFrame) -> pd.DataFrame:
    grid = pd.date_range(IS_START, IS_END - pd.Timedelta(hours=1), freq="h", tz="UTC")
    f = pd.DataFrame(index=grid)
    f["venue"] = "binance"
    f["market_type"] = "usdm_perp"
    src = df1h["fill_source"] if "fill_source" in df1h.columns else pd.Series("binance_um", index=df1h.index)
    f["source"] = src.reindex(grid).fillna("missing")
    f["is_missing"] = ~grid.isin(df1h.index)
    f["is_imputed"] = f["source"].ne("binance_um") & ~f["is_missing"]
    vol = df1h["volume"].reindex(grid)
    med = vol.shift(1).rolling(168, min_periods=24).median()
    f["volume_vs_trailing_median"] = vol / med
    f["is_partial"] = (f["volume_vs_trailing_median"] < PARTIAL_VOLUME_FRACTION).fillna(False)
    return f


def box_flags(boxes: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """每个箱体分布窗口 [window_start, window_end) 内的缺失 / 补数 / 疑似部分小时数。"""
    t = flags.index.as_unit("ns").asi8
    ws = pd.DatetimeIndex(boxes["window_start"]).as_unit("ns").asi8
    we = pd.DatetimeIndex(boxes["window_end"]).as_unit("ns").asi8
    a, b = np.searchsorted(t, ws, side="left"), np.searchsorted(t, we, side="left")
    cm = {k: np.concatenate([[0], np.cumsum(flags[k].to_numpy(dtype=np.int64))]) for k in ("is_missing", "is_imputed", "is_partial")}
    return pd.DataFrame({f"window_n_{k[3:]}": cm[k][b] - cm[k][a] for k in cm}, index=boxes.index)


DAILY_FEATURES = ("above_sma200", "sma200_up", "mom20d", "vol30d", "vol30d_pct", "dd365")


def spot_warmup_dependence_exact(df1h: pd.DataFrame, df4h: pd.DataFrame, spot: pd.Series, bars: np.ndarray) -> pd.DataFrame:
    """逐个日线状态特征精确判断是否依赖 2020 年之前的现货预热（审计复核 AF-8）：分别带预热与不带预热计算
    attribution.regime_features，取值不同（含一边为 NaN）即为依赖。0/1 特征按相等比较，连续特征按相对 1e-9 比较
    （滚动求和的浮点累积误差不算依赖）。另给出任一特征依赖预热的汇总列。"""
    import attribution as A
    a = A.regime_features(df1h, df4h, spot, bars)
    b = A.regime_features(df1h, df4h, None, bars)
    out = {}
    for k in DAILY_FEATURES:
        x, y = a[k].to_numpy(dtype=np.float64), b[k].to_numpy(dtype=np.float64)
        both_nan = np.isnan(x) & np.isnan(y)
        close = np.isclose(x, y, rtol=1e-9, atol=0.0)
        out[f"regime_{k}_uses_spot_warmup"] = ~(both_nan | close)
    df = pd.DataFrame(out)
    df["regime_any_daily_feature_uses_spot_warmup"] = df.any(axis=1)
    return df


def spot_warmup_dependence(signal_time: pd.Series) -> pd.DataFrame:
    """决策所用的最近一根已收盘日线为 D（signal_time 之前收盘），SMA200 窗口 [D−199, D]、365 日特征窗口 [D−364, D]；
    窗口起点早于 2020-01-01 即用到了现货预热收盘（只有价格）。"""
    d = (pd.DatetimeIndex(signal_time) - pd.Timedelta(days=1)).floor("D")
    return pd.DataFrame({"regime_sma200_uses_spot_warmup": (d - pd.Timedelta(days=199)) < IS_START,
                         "regime_365d_uses_spot_warmup": (d - pd.Timedelta(days=364)) < IS_START})


def summary(df1h: pd.DataFrame, df4h: pd.DataFrame, boxes: pd.DataFrame) -> dict:
    f = bar_flags(df1h)
    bf = box_flags(boxes, f)
    return {"hours": int(len(f)), "missing": int(f["is_missing"].sum()), "imputed": int(f["is_imputed"].sum()),
            "partial_flagged": [str(t) for t in f.index[f["is_partial"]]],
            "sources": f["source"].value_counts().to_dict(),
            "h4_incomplete": int((df4h["n_bars"] < 4).sum()), "h4_total": int(len(df4h)),
            "boxes_with_missing_or_imputed_hours": int(((bf["window_n_missing"] + bf["window_n_imputed"]) > 0).sum()),
            "boxes_with_partial_hours": int((bf["window_n_partial"] > 0).sum()), "boxes": int(len(boxes)),
            "partial_rule": f"volume < {PARTIAL_VOLUME_FRACTION:.0%} of the trailing 168-hour median (disclosure only)"}

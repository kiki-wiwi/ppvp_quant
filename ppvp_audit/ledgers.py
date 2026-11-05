"""
ppvp_audit/ledgers.py — 机会级与成交级账本、109 笔交易核对表（IS only）。

机会 = 状态机发出的每个原始信号（sig_raw，第 2 次测试中触及支撑带、收盘 ≥ 带下沿、且为看涨 Pinbar / 吞没的 K 线）。
每个机会的去向：filled（成交）、blocked_rr（VAH 不高于信号收盘，基线唯一的信号层过滤）、skipped_position_open
（持仓期间出现，未成交）、或实验版本自己的拒绝原因。未成交的机会也保留在机会表中，不从分母消失。

两次测试之间的“脱离”从状态序列复原：第 2 次测试开始（t2_start）之前紧邻的 DEPARTED（state = 3）连续段即为最终的
脱离段，其首根 d0 是第一根整根位于带上沿之上的 K 线（low > 带上沿），第 1 次测试最后一根带内 K 线 t1_last = d0 − 1；
状态机在 t2_start 用 max(high[d0 : t2_start]) − 带上沿(t2_start) ≥ 1·ATR，或 t2_start − t1_last ≥ 6 判定第 2 次测试。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import attribution as A
from config import IS_END, IS_START, BacktestConfig, StrategyConfig

from .common import load_is_market

NS_H = pd.Timedelta(hours=1)


def departure_diagnostics(sig: pd.DataFrame, df1h: pd.DataFrame, bars: np.ndarray, scfg: StrategyConfig) -> pd.DataFrame:
    st = sig["state"].to_numpy()
    h = df1h["high"].to_numpy(dtype=np.float64)
    zhi = sig["zone_hi"].to_numpy(dtype=np.float64)
    atr = sig["atr"].to_numpy(dtype=np.float64)
    t2s = sig["t2_start"].to_numpy(dtype=np.int64)
    rows = []
    for b in bars:
        s = int(t2s[b])
        d0 = s
        while d0 - 1 >= 0 and st[d0 - 1] == 3:
            d0 -= 1
        ok = d0 < s
        max_high = float(h[d0:s].max()) if ok else np.nan
        bounce = (max_high - zhi[s]) / atr[s] if ok else np.nan
        gap = s - (d0 - 1) if ok else np.nan
        by_bounce = bool(ok and bounce >= scfg.min_bounce_atr)
        by_bars = bool(ok and gap >= scfg.min_bars_between_touches)
        rows.append({"departure_bar": d0 if ok else -1, "t1_last_bar": d0 - 1 if ok else -1,
                     "departure_bars": s - d0 if ok else 0, "departure_max_high": max_high,
                     "departure_height_atr": bounce, "touch_gap_bars_from_t1_last": gap,
                     "second_test_qualified_by": ("bounce+bars" if by_bounce and by_bars else "bounce" if by_bounce
                                                  else "bars_only" if by_bars else "unexplained")})
    return pd.DataFrame(rows)


def box_lifecycle(boxes: pd.DataFrame, df1h: pd.DataFrame) -> pd.DataFrame:
    """每个箱体（profile 版本）的确认、首次可被 1H 使用与被下一个箱体取代的时刻。"""
    b = boxes.copy()
    first_use = np.searchsorted(df1h.index.asi8 if False else df1h.index.as_unit("ns").asi8,
                                pd.DatetimeIndex(b["available_time"]).as_unit("ns").asi8, side="left")
    b["available_at_1h"] = [df1h.index[i] if i < len(df1h) else pd.NaT for i in first_use]
    b["superseded_at"] = list(pd.DatetimeIndex(b["available_time"])[1:]) + [pd.NaT]
    return b


def opportunities(out: dict, scfg: StrategyConfig, bcfg: BacktestConfig, version: str = "B_original") -> pd.DataFrame:
    """全部原始信号（机会）及其决策时刻可知的特征与最终去向。"""
    df1h, sig, boxes, res = out["df1h"], out["signals"], out["boxes"], out["result"]
    _, _, spot = load_is_market()
    bars = np.flatnonzero(sig["sig_raw"].to_numpy())
    o, h, l, c = (df1h[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close"))
    slip = float(bcfg.slippage)
    bx = box_lifecycle(boxes, df1h)
    sbox = sig["box_id"].to_numpy(dtype=np.int64)
    t1b = sig["t1_bar"].to_numpy(dtype=np.int64)
    t2s = sig["t2_start"].to_numpy(dtype=np.int64)
    entry_ref = c[bars]
    entry_exp = entry_ref * (1.0 + slip)
    sl = sig["sl"].to_numpy()[bars]
    vah = sig["VAH"].to_numpy()[bars]
    val = sig["VAL"].to_numpy()[bars]
    poc = sig["POC"].to_numpy()[bars]
    atr = sig["atr"].to_numpy()[bars]
    df = pd.DataFrame({
        "version": version, "signal_bar": bars, "signal_time": df1h.index[bars] + NS_H,
        "order_time": df1h.index[bars] + NS_H, "order_type": "market_at_signal_close",
        "profile_id": sbox[bars], "profile_available_time": pd.DatetimeIndex(bx["available_time"].to_numpy()[sbox[bars]]),
        "profile_available_at_1h": pd.DatetimeIndex(bx["available_at_1h"].to_numpy()[sbox[bars]]),
        "profile_superseded_at": pd.DatetimeIndex(bx["superseded_at"].to_numpy()[sbox[bars]]),
        "pivot_a_time": pd.DatetimeIndex(bx["pivot_a_time"].to_numpy()[sbox[bars]]),
        "pivot_b_time": pd.DatetimeIndex(bx["pivot_b_time"].to_numpy()[sbox[bars]]),
        "pivot_pair": (bx["pivot_a_type"].astype(str) + bx["pivot_b_type"].astype(str)).to_numpy()[sbox[bars]],
        "window_start": pd.DatetimeIndex(bx["window_start"].to_numpy()[sbox[bars]]),
        "window_end": pd.DatetimeIndex(bx["window_end"].to_numpy()[sbox[bars]]),
        "t1_bar": t1b[bars], "t1_time": df1h.index[t1b[bars]], "t1_low": sig["t1_low"].to_numpy()[bars],
        "profile_id_at_t1": sbox[t1b[bars]],
        "t2_start_bar": t2s[bars], "t2_start_time": df1h.index[t2s[bars]], "t2_low": sig["t2_low"].to_numpy()[bars],
        "profile_id_at_t2_start": sbox[t2s[bars]],
        "pattern": sig["pattern"].to_numpy()[bars],
        "entry_ref_close": entry_ref, "entry_expected_fill": entry_exp,
        "VAL": val, "POC": poc, "VAH": vah, "SL": sl, "ATR": atr,
        "zone_lo": sig["zone_lo"].to_numpy()[bars], "zone_hi": sig["zone_hi"].to_numpy()[bars],
        "confirmation_high": h[bars], "confirmation_low": l[bars], "confirmation_range": h[bars] - l[bars],
        "confirmation_range_atr": (h[bars] - l[bars]) / atr,
        "stop_distance_pct": (entry_exp - sl) / entry_exp,
        "target_distance_pct": vah / entry_exp - 1.0,
        "RR_close": (vah - entry_ref) / (entry_ref - sl), "RR": (vah - entry_exp) / (entry_exp - sl),
        "normalized_entry_position": (entry_exp - val) / (vah - val),
        "entry_below_poc": entry_exp < poc,
        "box_width_atr": (vah - val) / atr,
        "next_open": np.where(bars + 1 < len(df1h), o[np.minimum(bars + 1, len(df1h) - 1)], np.nan),
    })
    df["next_open_vs_close_bp"] = (df["next_open"] / df["entry_ref_close"] - 1.0) * 1e4
    df["profile_changed_between_tests"] = (df["profile_id_at_t1"] != df["profile_id"]) | (df["profile_id_at_t2_start"] != df["profile_id"])
    df = pd.concat([df, departure_diagnostics(sig, df1h, bars, scfg)], axis=1)
    reg = A.regime_features(df1h, out["df4h"], spot, bars)
    reg.columns = [f"regime_{k}" if k != "daily_label" else "regime_asof_daily_label" for k in reg.columns]
    df = pd.concat([df, reg], axis=1)
    # 去向
    tradable = sig["signal"].to_numpy()[bars]
    status = np.where(sig["block_rr"].to_numpy()[bars], "blocked_rr_vah_not_above_close",
                      np.where(sig["block_width"].to_numpy()[bars], "blocked_width",
                               np.where(sig["block_regime"].to_numpy()[bars], "blocked_regime", "tradable")))
    tr = res.trades.set_index("entry_bar")
    filled = np.isin(bars, tr.index.to_numpy())
    status = np.where(tradable & filled, "filled", np.where(tradable & ~filled, "skipped_position_open", status))
    df["status"] = status
    for col_src, col in (("entry_time", "fill_time"), ("entry_px", "entry_fill"), ("exit_time", "exit_time"),
                         ("exit_px", "exit_fill"), ("exit_reason", "exit_reason"), ("qty", "qty"),
                         ("notional", "notional"), ("fees", "fees"), ("funding", "funding"), ("gross_pnl", "gross_pnl"),
                         ("pnl", "net_pnl"), ("r_multiple", "net_R"), ("hold_bars", "hold_bars")):
        df[col] = tr[col_src].reindex(bars).to_numpy()
    return df


def trade_ledger(out: dict, scfg: StrategyConfig, bcfg: BacktestConfig, df1m: pd.DataFrame | None,
                 version: str = "B_original") -> pd.DataFrame:
    """成交级账本：成本分解（毛盈亏 / 手续费 / 滑点影响 / 资金费 / 净盈亏）、R 的分母、账户口径，以及 1m 路径核对。"""
    res, df1h = out["result"], out["df1h"]
    tf = A.trade_frame(res, df1h, out["boxes"], scfg)
    slip = float(bcfg.slippage)
    ep, xp, q = tf["entry_px"].to_numpy(), tf["exit_px"].to_numpy(), tf["qty"].to_numpy()
    tf["version"] = version
    tf["entry_ref"] = ep / (1 + slip)
    tf["exit_ref"] = xp / (1 - slip)
    tf["gross_pnl_ref"] = q * (tf["exit_ref"] - tf["entry_ref"])          # 成本前（参考价）毛盈亏
    tf["slippage_impact"] = tf["gross_pnl_ref"] - tf["gross_pnl"]        # 已体现在成交价中，不再另扣
    tf["commission"] = tf["fees"]
    tf["funding_cashflow"] = tf["funding"]
    tf["net_pnl"] = tf["pnl"]
    assert np.allclose(tf["gross_pnl"] - tf["commission"] - tf["funding_cashflow"], tf["net_pnl"], rtol=0, atol=1e-6)
    tf["initial_risk_usd"] = q * (ep - tf["sl"].to_numpy())
    tf["gross_R_ref"] = tf["gross_pnl_ref"] / tf["initial_risk_usd"]
    tf["net_R"] = tf["net_pnl"] / tf["initial_risk_usd"]
    tf["risk_pct_of_equity"] = tf["initial_risk_usd"] / tf["equity_at_entry"]
    tf["return_on_notional"] = tf["net_pnl"] / tf["notional"]
    tf["return_on_equity"] = tf["net_pnl"] / tf["equity_at_entry"]
    tf["hold_hours"] = tf["hold_bars"]
    if df1m is not None:
        tf = tf.join(path_check_1m(tf, df1m))
    return tf


def path_check_1m(tf: pd.DataFrame, df1m: pd.DataFrame) -> pd.DataFrame:
    """离场 K 线内的 1m 路径：首次触及 SL / TP / POC 的分钟，按 1m 路径的离场原因（同一分钟同时触及 SL 与 TP 记为 ambiguous）。"""
    rows = []
    lo_, hi_ = df1m["low"].to_numpy(), df1m["high"].to_numpy()
    t_ns = df1m.index.as_unit("ns").asi8
    for _, r in tf.iterrows():
        t0 = pd.Timestamp(r["exit_bar_open"]).as_unit("ns").value
        a = int(np.searchsorted(t_ns, t0))
        b = a + 60
        lo, hi = lo_[a:b], hi_[a:b]
        f_sl = np.flatnonzero(lo <= r["sl"])
        f_tp = np.flatnonzero(hi >= r["tp"])
        f_poc = np.flatnonzero(hi >= r["poc"]) if r["poc"] > r["entry_px"] else np.array([], int)
        m_sl = int(f_sl[0]) if len(f_sl) else -1
        m_tp = int(f_tp[0]) if len(f_tp) else -1
        if m_sl >= 0 and (m_tp < 0 or m_sl < m_tp):
            reason = "SL"
        elif m_tp >= 0 and (m_sl < 0 or m_tp < m_sl):
            reason = "TP"
        elif m_sl >= 0 and m_sl == m_tp:
            reason = "ambiguous_same_minute"
        else:
            reason = "none_in_exit_bar"
        rows.append({"exit_minute_sl": m_sl, "exit_minute_tp": m_tp, "exit_minute_poc": int(f_poc[0]) if len(f_poc) else -1,
                     "exit_reason_1m_path": reason, "exit_reason_matches_1m": reason == r["exit_reason"]})
    return pd.DataFrame(rows, index=tf.index)


def reconciliation(tl: pd.DataFrame) -> pd.DataFrame:
    """109 笔逐笔核对：离场类别、是否进入图 1 左 / 右图、未进入的原因、离场 K 线的路径歧义。"""
    r = pd.DataFrame({"entry_time": tl["entry_time"], "exit_time": tl["exit_time"], "exit_reason": tl["exit_reason"],
                      "entry_bar": tl["entry_bar"], "exit_bar": tl["exit_bar"],
                      "full_bars_between_entry_and_exit": (tl["exit_bar"] - tl["entry_bar"] - 1).clip(lower=0),
                      "poc_above_entry": tl["poc_above_entry"], "reached_poc_strict": tl["reached_poc_strict"]})
    sl = tl["exit_reason"] == "SL"
    tp = tl["exit_reason"] == "TP"
    r["in_fig1_left_panel"] = sl & tl["poc_above_entry"] & tl["mfe_r_strict"].notna()
    r["in_fig1_right_panel"] = tp & tl["mae_r_strict"].notna()
    why = np.where(sl & ~tl["poc_above_entry"], "SL: POC not above entry (no POC target to reach)",
          np.where(sl & tl["poc_above_entry"] & tl["mfe_r_strict"].isna(), "SL: exit on the first bar after entry (no full bar)",
          np.where(tp & tl["mae_r_strict"].isna(), "TP: exit on the first bar after entry (no full bar)", "")))
    r["excluded_from_fig1_because"] = why
    r["exit_bar_touches_tp_and_sl"] = tl["ambiguous_exit_bar"]
    for k in ("exit_reason_1m_path", "exit_reason_matches_1m", "exit_minute_sl", "exit_minute_tp", "exit_minute_poc"):
        if k in tl.columns:
            r[k] = tl[k]
    if "exit_minute_poc" in tl.columns:
        r["poc_reached_in_exit_bar_before_sl_1m"] = sl & tl["poc_above_entry"] & (tl["exit_minute_poc"] >= 0) & \
            ((tl["exit_minute_poc"] < tl["exit_minute_sl"]) | (tl["exit_minute_sl"] < 0))
    r["entry_and_exit_same_bar"] = tl["exit_bar"] == tl["entry_bar"]
    return r

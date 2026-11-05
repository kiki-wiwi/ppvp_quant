"""
ppvp_audit/slices.py — 可复现的交易切片（最多 16 笔）、最差月份 13 笔核查表、止损后走势与低盈亏比来源的描述统计。

抽样规则在运行前写定（种子 = SEED + 1）：按下列顺序逐类抽取，已选交易不重复；某类不足时全部列出。
1. 初始 RR < 0.5 的 3 笔：按该类中止盈 / 止损的笔数比例分配，但每个非空结果至少 1 笔。
2. 日线收盘在 SMA200 上方、仍以止损离场的 3 笔。
3. 初始 RR ≥ 2 的止盈 2 笔、止损 2 笔。
4. 止损前（严格口径）已到达 POC 的止损交易 2 笔。
5. 其余尚未选中的交易中随机 4 笔。
这些都是描述性的个案核查，不用于生成任何规则；最差月份 13 笔是事后归因，不能据此生成日期禁交易规则。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import FIG, OUT, SEED

H = pd.Timedelta(hours=1)
REASON_ZH = {"RR<0.5 (stop-loss)": "初始 RR < 0.5，止损", "RR<0.5 (take-profit)": "初始 RR < 0.5，止盈",
             "above daily SMA200, stopped out": "日线在 SMA200 上方仍止损", "RR>=2 (take-profit)": "初始 RR ≥ 2，止盈",
             "RR>=2 (stop-loss)": "初始 RR ≥ 2，止损", "reached POC, then stopped out": "先到 POC 后止损",
             "random from the rest": "其余交易中随机抽取"}


def select_slices(tl: pd.DataFrame, seed: int = SEED + 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    why: dict[int, str] = {}

    def take(pool_idx, k, label):
        pool = [i for i in pool_idx if i not in chosen]
        if len(pool) <= k:
            pick = pool
        else:
            pick = list(rng.choice(pool, size=k, replace=False))
        for i in sorted(pick):
            chosen.append(int(i))
            why[int(i)] = label
        return len(pick)

    rr = tl["rr_fill"]
    low = tl.index[rr < 0.5]
    low_tp = [i for i in low if tl.at[i, "exit_reason"] == "TP"]
    low_sl = [i for i in low if tl.at[i, "exit_reason"] == "SL"]
    n_low = min(3, len(low))
    k_sl = 0 if not low_sl else max(1, int(round(n_low * len(low_sl) / len(low))))
    k_sl = min(k_sl, len(low_sl), n_low - (1 if low_tp else 0))
    take(low_sl, k_sl, "RR<0.5 (stop-loss)")
    take(low_tp, n_low - k_sl, "RR<0.5 (take-profit)")
    take(tl.index[(tl["exit_reason"] == "SL") & (tl["above_sma200"] == 1.0)], 3, "above daily SMA200, stopped out")
    take(tl.index[(rr >= 2) & (tl["exit_reason"] == "TP")], 2, "RR>=2 (take-profit)")
    take(tl.index[(rr >= 2) & (tl["exit_reason"] == "SL")], 2, "RR>=2 (stop-loss)")
    take(tl.index[(tl["exit_reason"] == "SL") & tl["reached_poc_strict"].astype(bool)], 2, "reached POC, then stopped out")
    take(tl.index, 4, "random from the rest")
    sel = tl.loc[chosen].copy()
    sel.insert(0, "slice_reason", [why[i] for i in chosen])
    sel.insert(0, "slice_no", np.arange(1, len(sel) + 1))
    return sel


def post_stop_paths(tl: pd.DataFrame, df1h: pd.DataFrame, horizons=(6, 12, 24)) -> pd.DataFrame:
    """全部止损单：离场后固定 6 / 12 / 24 小时内是否到达原 POC / VAH，以及途中进一步下跌多少（描述统计，不用于改止损）。"""
    h = df1h["high"].to_numpy(dtype=np.float64)
    l = df1h["low"].to_numpy(dtype=np.float64)
    n = len(df1h)
    rows = []
    for _, r in tl[tl["exit_reason"] == "SL"].iterrows():
        xb = int(r["exit_bar"])
        risk = r["entry_px"] - r["sl"]
        row = {"entry_time": r["entry_time"], "exit_time": r["exit_time"], "poc_above_entry": bool(r["poc_above_entry"])}
        for H_ in horizons:
            a, b = xb + 1, min(xb + 1 + H_, n)
            row[f"window_complete_{H_}h"] = (xb + 1 + H_) <= n
            hh = h[a:b].max() if b > a else np.nan
            ll = l[a:b].min() if b > a else np.nan
            row[f"reached_poc_{H_}h"] = bool(r["poc_above_entry"] and hh >= r["poc"])
            row[f"reached_vah_{H_}h"] = bool(hh >= r["tp"])
            row[f"further_drop_below_sl_pct_{H_}h"] = float(min(0.0, ll / r["sl"] - 1.0))
            row[f"further_drop_below_sl_R_{H_}h"] = float(min(0.0, (ll - r["sl"]) / risk))
        rows.append(row)
    return pd.DataFrame(rows)


def low_rr_decomposition(tl: pd.DataFrame) -> dict:
    """RR = 目标距离 / 止损距离。log RR 的截面方差按 cov(log RR, log 目标) 与 −cov(log RR, log 止损) 分解（两者之和为 1），
    并给出 RR < 1 与 RR ≥ 1 两组的各项中位数。描述性，不预设唯一病因。"""
    e = tl["entry_px"]
    tgt = (tl["tp"] - e) / tl["atr"]
    stp = (e - tl["sl"]) / tl["atr"]
    ok = tgt > 0
    lr, lt, ls = np.log(tgt[ok] / stp[ok]), np.log(tgt[ok]), np.log(stp[ok])
    v = lr.var(ddof=1)
    out = {"n_rr_lt_1": int((tl["rr_fill"] < 1).sum()), "n_rr_ge_1": int((tl["rr_fill"] >= 1).sum()),
           "n_target_not_above_entry": int((~ok).sum()),
           "share_var_logRR_from_target": float(np.cov(lr, lt)[0, 1] / v),
           "share_var_logRR_from_stop": float(-np.cov(lr, ls)[0, 1] / v)}
    grp = {}
    for name, m in (("RR<1", tl["rr_fill"] < 1), ("RR>=1", tl["rr_fill"] >= 1)):
        s = tl[m]
        grp[name] = {
            "n": int(m.sum()), "median_target_atr": float(tgt[m].median()), "median_stop_atr": float(stp[m].median()),
            "median_box_width_atr": float(((s["VAH"] - s["VAL"]) / s["atr"]).median()),
            "median_normalized_entry_position": float(((s["entry_px"] - s["VAL"]) / (s["VAH"] - s["VAL"])).median()),
            "share_entry_above_band_top": float(s["entry_above_band"].mean()),
            "share_entry_above_poc": float((s["entry_px"] > s["POC"]).mean()),
            "median_confirmation_range_atr": float(s["confirmation_range_atr"].median()) if "confirmation_range_atr" in s else None,
            "share_profile_changed_since_first_test_start": float(s["profile_changed_since_first_test_start"].mean())
            if "profile_changed_since_first_test_start" in s else None,
            "share_profile_changed_since_chain_start": float(s["profile_changed_since_chain_start"].mean())
            if "profile_changed_since_chain_start" in s else None,
            "median_pattern_lag_bars": float(s["pattern_lag_bars"].median()),
            "share_tp_exit": float((s["exit_reason"] == "TP").mean()), "mean_net_R": float(s["r_multiple"].mean()),
        }
    out["groups"] = grp
    return out


def plot_slice(r: pd.Series, df1h: pd.DataFrame, df4h: pd.DataFrame, sig: pd.DataFrame, boxes: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from .plotstyle import BASE, C1, C2, C3, GRID, INK2, MUTED, SURF, style
    style()
    sig_t = pd.Timestamp(r["entry_time"])                 # 信号 K 线收盘 = 下单与成交时刻（基线按收盘价市价成交）
    exit_t = pd.Timestamp(r["exit_time"])
    box = boxes.loc[int(r["box_id"])]
    t1 = df1h.index[int(r["t1_bar"])]                     # 第 1 次测试最低点所在的 K 线
    t_fs = pd.Timestamp(r["first_test_start_time"]) if "first_test_start_time" in r else t1
    t0_1h = min(t1, t_fs, df1h.index[int(r["t2_start"])]) - 36 * H
    t9_1h = exit_t + 18 * H
    t0_4h = pd.Timestamp(box["window_start"]) - 8 * 4 * H
    fig, (ax4, ax1) = plt.subplots(2, 1, figsize=(10.5, 9.0), gridspec_kw={"height_ratios": [1, 1.25]})

    def candles(ax, d, width):
        up = d["close"] >= d["open"]
        for tt, row, u in zip(d.index, d.itertuples(), up):
            x = tt + width / 2
            ax.vlines(x, row.low, row.high, color=INK2 if u else MUTED, linewidth=0.8, zorder=2)
            ax.add_patch(Rectangle((tt + width * 0.15, min(row.open, row.close)), width * 0.7,
                                   max(abs(row.close - row.open), 1e-9), facecolor=SURF if u else MUTED,
                                   edgecolor=INK2 if u else MUTED, linewidth=0.6, zorder=3))
    d4 = df4h[(df4h.index >= t0_4h) & (df4h.index <= t9_1h)]
    candles(ax4, d4, 4 * H)
    for col, lab, colr, ls in (("pivot_a_time", "枢轴 a 发生", BASE, ":"), ("pivot_b_time", "枢轴 b 发生", BASE, ":")):
        ax4.axvline(pd.Timestamp(box[col]) + 2 * H, color=colr, linestyle=ls, linewidth=1.2)
    ax4.axvline(pd.Timestamp(box["available_time"]), color=C3, linewidth=1.4, label="箱体确认、可被 1H 使用（R=5 根 4H 后）")
    ax4.axvspan(pd.Timestamp(box["window_start"]), pd.Timestamp(box["window_end"]), color=GRID, alpha=0.5, zorder=0,
                label="成交量分布窗口 [枢轴 a 开盘, 枢轴 b 收盘)")
    for lvl, lab, colr in (("VAL", "VAL", C1), ("POC", "POC", MUTED), ("VAH", "VAH", C2)):
        ax4.hlines(box[lvl], pd.Timestamp(box["available_time"]), t9_1h, color=colr, linewidth=1.3, linestyles="-")
        ax4.text(t9_1h, box[lvl], f" {lab}", color=colr, va="center", fontsize=8)
    ax4.axvspan(sig_t, t9_1h, color=GRID, alpha=0.35, zorder=0)
    reason = REASON_ZH.get(r["slice_reason"], r["slice_reason"])
    ax4.set_title(f"切片 {int(r['slice_no'])}：{reason}；4H（信号所用箱体 {int(r['box_id'])} 的分布窗口与确认时刻）")
    ax4.legend(loc="upper left", bbox_to_anchor=(0.0, -0.12), fontsize=7.5, ncol=2)

    d1 = df1h[(df1h.index >= t0_1h) & (df1h.index <= t9_1h)]
    candles(ax1, d1, H)
    s1 = sig.loc[d1.index]
    ax1.fill_between(s1.index + H / 2, s1["zone_lo"], s1["zone_hi"], step="mid", color=C1, alpha=0.12, linewidth=0,
                     label="支撑带 [VAL−0.5ATR, VAL+0.5ATR]（决策时已知）")
    ax1.axvline(t1 + H / 2, color=C1, linestyle="--", linewidth=1.1, label="第 1 次测试低点所在 K 线")
    if t_fs != t1:
        ax1.axvline(t_fs + H / 2, color=C1, linestyle=(0, (1, 3)), linewidth=1.1,
                    label=f"第 1 次测试开始（roll {int(r.get('n_rolls', 0))} 次）")
    b1 = int(r.get("profile_id_at_first_test_start", r["box_id"]))
    if b1 != int(r["box_id"]):
        v1 = float(boxes.loc[b1, "VAL"])
        ax1.hlines(v1, t_fs, sig_t, color=C1, linestyle="--", linewidth=1.0,
                   label=f"第 1 次测试开始时的箱体 {b1} 的 VAL（跨了箱体版本）")
    if int(r.get("departure_bar", -1)) >= 0:
        ax1.axvline(df1h.index[int(r["departure_bar"])] + H / 2, color=C3, linestyle=":", linewidth=1.1,
                    label=f"脱离：整根 K 线高于带上沿（反弹 {r['departure_height_atr']:.2f} ATR）")
    ax1.axvline(df1h.index[int(r["t2_start"])] + H / 2, color=C1, linestyle="-.", linewidth=1.1, label="第 2 次测试开始")
    ax1.axvline(sig_t, color=INK2, linewidth=1.4, label="确认 K 线收盘：下单并按收盘价 + 滑点成交")
    for lvl, lab, colr in (("sl", "SL", C2), ("poc", "POC", MUTED), ("tp", "VAH 止盈", C3)):
        ax1.hlines(r[lvl], sig_t, exit_t, color=colr, linewidth=1.4)
        ax1.text(exit_t, r[lvl], f" {lab}", color=colr, va="center", fontsize=8)
    ax1.plot([sig_t], [r["entry_px"]], marker="^", color=INK2, markersize=8, zorder=5)
    ax1.plot([exit_t - H / 2], [r["exit_px"]], marker="v", color=C2 if r["exit_reason"] == "SL" else C3, markersize=8, zorder=5)
    ax1.axvspan(sig_t, t9_1h, color=GRID, alpha=0.35, zorder=0, label="灰底：决策之后的事后行情")
    ax1.set_title(f"1H：入场 {sig_t:%Y-%m-%d %H:%M}，离场 {exit_t:%Y-%m-%d %H:%M}（{r['exit_reason']}），"
                  f"RR {r['rr_fill']:.2f}，净 R {r['r_multiple']:+.2f}")
    ax1.legend(loc="upper left", bbox_to_anchor=(0.0, -0.1), fontsize=7.2, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

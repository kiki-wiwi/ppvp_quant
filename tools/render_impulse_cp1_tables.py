#!/usr/bin/env python3
"""
tools/render_impulse_cp1_tables.py — 把 reports/impulse_vs_box_round/cp1/cp1_results.json 渲染为 Markdown 表格
（reports/impulse_vs_box_round/cp1/cp1_tables.md）。阶段记录 docs/phase_logs/IMPULSE_CP1_data.md 中的表格由此逐字生成。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CP1 = ROOT / "reports" / "impulse_vs_box_round" / "cp1"
YEARS = ("2020", "2021", "2022", "2023", "IS")
YL = {"2020": "2020", "2021": "2021", "2022": "2022", "2023": "2023", "IS": "IS 合计"}


def n(x) -> str:
    """数字加千分位；单元格内的竖线转义，避免破坏 Markdown 表格。"""
    return (f"{x:,}" if isinstance(x, int) and not isinstance(x, bool) else str(x)).replace("|", "\\|")


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(n(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x) -> str:
    return "∞" if x == "inf" else f"{x * 100:.2f}%"


def render(r: dict) -> str:
    s = []
    dl = r["download"]
    names = {"perp_1m": "永续 1m 月度（主数据）", "perp_15m_official": "永续 15m 月度（仅核验）",
             "perp_1d_official": "永续 1d 月度（仅核验）", "spot_1d_warmup": "现货 1d 2019-09…12（ATR 预热）",
             "spot_1m_warmup": "现货 1m 2019-11…12（仅 S4 敏感性）"}
    s.append("### 表 1　月度归档下载与 SHA256 校验\n")
    s.append(table(["数据集", "文件数", "行数", "SHA256 与官方 CHECKSUM 一致", "带表头的文件（自）"],
                   [[names[k], dl[k]["files"], dl[k]["rows"], "全部一致" if dl[k]["all_checksum_verified"] else "否",
                     f'{dl[k]["files_with_header"]}（{dl[k]["first_with_header"]}）' if dl[k]["first_with_header"] else "0"]
                    for k in names]))
    # 额外的日度归档（来源记录）
    adj = dl["adjudicated_daily_archives"][0]
    extra = {"裁决日 2023-11-10（日度 1m + 逐笔）": 2,
             "裁决对照日 2023-11-09 / 11（日度 1m + 逐笔）": 2 * len(adj.get("control_days_background", [])),
             "成交量差异定位（逐笔）": len(r["qa_volume_discrepancies_localized"]["days"]),
             "月度 vs 日度 1m 逐位比对（日度 1m）": len(r["qa_monthly_vs_daily_archive"]),
             "维护停机窗口核对（逐笔）": len(r["qa_maintenance_windows_vs_aggtrades"]),
             "高波动对照日 2020-03-12 / 2021-05-19（日度 1m + 逐笔）": 2 * len(r["qa_kline_vs_aggtrades_background"])}
    s.append("\n额外下载的官方日度归档（每个都校验 SHA256 并在结果 JSON 中记录 URL 与 SHA256）：" +
             "；".join(f"{k} {v} 个" for k, v in extra.items()) + "。")
    q = r["qa_integrity_1m"]
    s.append("\n### 表 2　1m 数据完整性（IS 2020-01-01 00:00 至 2023-12-31 23:59）\n")
    s.append(table(["检查项", "结果"], [
        ["行数 / 应有分钟数", f'{q["rows"]:,} / {q["expected_rows"]:,}'],
        ["缺失分钟 / 重复时间戳 / 不在分钟网格上的行", f'{q["missing_minutes"]} / {q["duplicates"]} / {q["off_grid_rows"]}'],
        ["OHLC 逻辑错误 / 非正价格 / 负成交量 / 空值", f'{q["bad_ohlc_rows"]} / {q["non_positive_price_rows"]} / '
                                               f'{q["negative_volume_rows"]} / {q["nan_cells"]}'],
        ["原始 close_time ≠ open_time + 59,999 ms 的行", q["close_time_mismatch_rows"]],
        ["主动买量 > 成交量 / 成交均价落在高低价之外", f'{q["taker_buy_gt_volume_rows"]} / {q["vwap_outside_high_low_rows"]}'],
        ["零成交分钟（其中维护占位：O=H=L=C；占位价等于前一分钟收盘）",
         f'{q["zero_volume_minutes"]}（{q["placeholder_minutes"]}；{q["placeholder_minutes_priced_at_previous_close"]}）'],
        ["零成交分钟分布", "；".join(f"{a[:16]} 至 {b[11:16]}（{c} 分钟）" for a, b, c in q["placeholder_ranges"])],
        ["|1m 对数收益| 最大值", f'{q["max_abs_1m_log_return"] * 100:.2f}%（{q["max_abs_1m_log_return_at"][:16]}）'],
        ["|1m 收益| > 2% 的分钟", q["minutes_abs_1m_return_gt_2pct"]],
        ["来源：月度归档 / 日度归档替换（分钟）", f'{q["rows_by_src"]["0"]:,} / {q["rows_by_src"]["1"]:,}'],
    ]))
    mw = r["qa_maintenance_windows_vs_aggtrades"]
    s.append("\n维护停机核对：" + "；".join(
        f'{m["window"][0][:16]} 起 {m["window"][2]} 分钟，逐笔归档窗口内成交 {m["agg_trades_inside_window"]} 笔，'
        f'全天成交量 1m {m["day_volume_1m"]:,.3f} / 逐笔 {m["day_volume_aggtrades"]:,.3f}' for m in mw) + "。")
    s.append("\n### 表 3　跨周期交叉对齐（1m 按 UTC 边界聚合后逐根比对；成交量按绝对差 > 1e-6 BTC 判定）\n")
    rows = []
    for key, lab in (("qa_1m_vs_repo_1h", "仓库 1H（IS 行）"), ("qa_1m_vs_official_15m", "官方 15m 归档"),
                     ("qa_1m_vs_official_1d", "官方 1d 归档")):
        x = r[key]
        nb = x["ohlc_mismatch_bars"]
        days = sorted({t[:10] for t in x["ohlc_mismatch_bars_list"]})
        rows.append([lab, x["bars_common"], f'{nb}' + (f'（{"、".join(days)}）' if nb else ""),
                     x["volume_mismatch_bars"], x["volume_reference_zero_aggregate_positive_bars"],
                     f'{x["volume_max_abs_diff"]:,.3f}', pct(x["volume_max_rel_diff"]), x["count_mismatch_bars"],
                     x.get("reference_close_time_mismatch_rows", "—")])
    s.append(table(["参照", "共同 K 线", "OHLC 不一致的 K 线", "成交量不一致", "其中参照为 0", "成交量最大绝对差（BTC）",
                    "最大相对差", "成交笔数不等", "参照 close_time 异常"], rows))
    a = adj
    s.append("\n### 表 4　2023-11-10 三方裁决证据\n")
    st = a["silent_traded_minutes"]
    s.append(table(["证据", "月度 1m", "日度 1m", "逐笔聚合（aggTrades）", "仓库 1H"], [
        ["全天成交量（BTC）", f'{a["day_volume"]["monthly"]:,.3f}', f'{a["day_volume"]["daily"]:,.3f}',
         f'{a["day_volume"]["aggtrades"]:,.3f}', f'{a["day_volume"]["repo_1h"]:,.3f}'],
        ["零成交但逐笔有成交的分钟", f'{st["monthly"]}（{st["monthly_range"][0][11:16]}–{st["monthly_range"][1][11:16]}，'
                                 f'逐笔成交 {st["aggtrades_volume_in_monthly_silent_minutes"]:,.3f} BTC）', st["daily"], "—", "—"],
        ["逐笔归档相对日度 1m 的空洞分钟", "—", a["aggtrades_holes_vs_daily"]["minutes"], "—", "—"],
        ["1H 聚合与仓库 1H 的 OHLC 不一致", "、".join(h[11:16] for h in a["hours_mismatch_monthly_vs_1h"]),
         len(a["hours_mismatch_daily_vs_1h"]), len(a["hours_mismatch_aggtrades_vs_1h"]), "—"],
        ["规则 v1（逐分钟完全一致）", "—", "未通过" if not a["rule_v1_strict_pass"] else "通过", "—", "—"],
        ["规则 v2 (a)/(b)/(c)", "违反 (a)(c)" if a["rule_v2"]["monthly_violates_a_or_c"] else "—",
         "/".join("通过" if a["rule_v2"][k] else "否" for k in ("a_daily_1h_ohlc_equal",
                                                              "b_aggtrades_complete_and_day_volume_equal",
                                                              "c_daily_no_silent_traded_minutes")), "—", "—"],
    ]))
    bg = a.get("control_days_background", []) + r["qa_kline_vs_aggtrades_background"]
    s.append("\n### 表 4b　K 线与逐笔聚合的背景差异（月度与日度 1m 逐位相同的对照日）\n")
    s.append(table(["对照日", "月度 = 日度（逐位）", "OHLC 不一致分钟", "成交量不一致分钟", "全天成交量 K 线 / 逐笔（BTC）",
                    "逐笔空洞（K 线有成交、逐笔无记录）", "逐笔成交 id 缺口"], [
        [c["day"], "是" if c["monthly_equals_daily_bitwise"] else "否", c["kline_vs_aggtrades_ohlc_mismatch_minutes"],
         c["kline_vs_aggtrades_volume_mismatch_minutes"], f'{c["day_volume_kline"]:,.3f} / {c["day_volume_aggtrades"]:,.3f}',
         (f'{c["kline_traded_minutes_absent_in_aggtrades"]} 分钟（' + "、".join(f"{x[0][11:16]}–{x[1][11:16]}" for x in
                                                                        c["aggtrades_hole_ranges"]) +
          f'，K 线成交 {c["kline_volume_in_aggtrades_holes"]:,.3f} BTC）') if c["kline_traded_minutes_absent_in_aggtrades"] else "0",
         f'{c["aggtrades_archive"].get("trade_id_gaps", "—"):,}' if isinstance(c["aggtrades_archive"].get("trade_id_gaps"), int)
         else "—"] for c in bg]))
    md = r["qa_monthly_vs_daily_archive"]
    s.append(f"\n月度 vs 日度 1m 逐位比对：{len(md)} 个自然日中 {sum(x['bitwise_equal'] for x in md)} 个逐位相同"
             + ("" if all(x["bitwise_equal"] for x in md) else "（不同：" + "、".join(x["day"] for x in md if not x["bitwise_equal"]) + "）")
             + "（2023-11-10 的缓存即日度归档）。")
    loc = r["qa_volume_discrepancies_localized"]
    s.append("\n### 表 5　1m 与 1H / 官方 1d 成交量不一致的自然日（逐笔归档逐分钟定位）\n")
    rows = []
    for d in loc["days"]:
        v = d["day_volume"]
        eq = {"repo_1h": "1H", "official_1d": "1d", "1m": "1m"}
        rows.append([d["day"], f'{v["1m"]:,.3f}', f'{v["repo_1h"] - v["1m"]:+,.3f}', f'{v["official_1d"] - v["1m"]:+,.3f}',
                     f'{v["aggtrades"] - v["1m"]:+,.3f}', "、".join(eq[k] for k in d["products_equal_to_aggtrades"]) or "无",
                     "、".join(t[11:16] for t in d["tail_omission_minutes"]) or "无", d["single_trade_displacement_minutes"]])
    s.append(table(["自然日", "1m 全天（BTC）", "1H − 1m", "官方 1d − 1m", "逐笔 − 1m", "与逐笔相等的产品",
                    "分钟末尾漏记", "单笔跨分钟错位（分钟数）"], rows))
    s.append("\n### 表 5b　分钟末尾漏记（K 线为该分钟逐笔序列的前缀，截掉末尾的成交）\n")
    rows = []
    for m in loc["minutes"]:
        if m["class"] != "tail_omission":
            continue
        t = m["tail_drop"]
        rows.append([m["minute"][:16], f'{t["dropped_agg_trades"]:,}', f'{t["dropped_volume"]:,.3f}',
                     f'{t["dropped_ms_in_minute"][0] / 1000:.3f}–{t["dropped_ms_in_minute"][1] / 1000:.3f}',
                     f'{m["kline_hlc"][2]:,} → {m["aggtrades_hlc"][2]:,}',
                     (f'高 {m["kline_hlc"][0]:,} → {m["aggtrades_hlc"][0]:,}' if m["kline_hlc"][0] != m["aggtrades_hlc"][0] else "")
                     + (f'低 {m["kline_hlc"][1]:,} → {m["aggtrades_hlc"][1]:,}' if m["kline_hlc"][1] != m["aggtrades_hlc"][1] else "")
                     or "相同", "是" if t["kline_is_exact_prefix"] else f'否（残差 {t["prefix_volume_residual"]:+.3f}）'])
    s.append(table(["分钟", "被截掉的逐笔聚合笔数", "被截掉的成交量（BTC）", "截掉部分在分钟内的时刻（秒）",
                    "收盘价 K 线 → 逐笔", "最高 / 最低价 K 线 → 逐笔", "恰为前缀"], rows))
    ex = r["tail_omission_exposure"]
    s.append("\n这些分钟与触发的时间关系（只看时间戳）：回看窗口 [T−15, T] 含漏记分钟的触发分钟 Up "
             f'{ex["up"]["trigger_minutes_lookback_covers_omission"]} / Down {ex["dn"]["trigger_minutes_lookback_covers_omission"]}'
             f'（其中被 sched45 接受的 {ex["up"]["sched45_lookback_covers_omission"]} / {ex["dn"]["sched45_lookback_covers_omission"]}），'
             f'最长持仓窗口 [T+1, T+30] 含漏记分钟的触发分钟 Up {ex["up"]["trigger_minutes_max_holding_covers_omission"]} / '
             f'Down {ex["dn"]["trigger_minutes_max_holding_covers_omission"]}（被 sched45 接受的 '
             f'{ex["up"]["sched45_max_holding_covers_omission"]} / {ex["dn"]["sched45_max_holding_covers_omission"]}）。')
    d = r["qa_daily_available_time"]
    s.append("\n### 表 6　UTC 日线可用时间\n")
    s.append(table(["检查项", "结果"], [
        ["日线根数 / 首日 / 末日", f'{d["days"]:,} / {d["first_day"][:10]} / {d["last_day"][:10]}'],
        ["开盘时间不在 UTC 00:00 / 日期不连续", f'{d["open_time_not_midnight_utc"]} / {d["consecutive_days_gap"]}'],
        ["由 1m 独立核对：首根 1m 开盘 ≠ 日线 open_time 的天数", d["days_first_1m_open_ne_day_open"]],
        ["由 1m 独立核对：末根 1m 收盘 ≠ open_time + 24h 的天数", d["days_last_1m_close_ne_available_time"]],
        ["由 1m 独立核对：不是恰好 1,440 根 1m 的天数", d["days_with_ne_1440_minutes"]],
        ["最后一根日线的 available_time（≤ IS_END）", f'{d["last_available_time"][:19]}（{d["last_available_time_le_is_end"]}）'],
        ["官方 1d：close_time = open_time + 24h − 1ms 的行", f'{d["official_1d_close_time_eq_open_plus_24h_minus_1ms"]:,} / '
                                                    f'{d["official_1d_rows"]:,}'],
    ]))
    sp = r["qa_spot_warmup"]
    s.append("\n现货预热：1d 共 " + f'{sp["spot_1d_rows"]} 根（{sp["spot_1d_first"][:10]} 至 {sp["spot_1d_last"][:10]}，'
             f'缺口 {sp["spot_1d_gaps"]}），与仓库现货 1H 聚合日线 OHLC 不一致 {len(sp["spot_1d_vs_repo_spot_1h_ohlc_mismatch_days"])} 天；'
             f'1m 共 {sp["spot_1m_rows"]:,} 根，缺失 {sp["spot_1m_missing_minutes"]} 分钟（' +
             "；".join(f"{x[0][:16]} 至 {x[1][11:16]}" for x in sp["spot_1m_missing_ranges"]) + "）。")
    pt = r["prefix_truncation"]
    s.append("\n### 表 6b　前缀截断核验（只用截断点之前的 1m 数据重算，与全量结果逐位比对）\n")
    s.append(table(["截断点", "比对分钟数", "特征与维护污染", "状态表（9 个口径）", "逐分钟掩码（相同 / 总数）",
                    "截断日当天截断前的箱体内分钟", "植入泄漏是否检出（状态表 / 只看价位）"], [
        [p["cut"][:16], p["minutes_compared"], "逐位相同" if p["features_and_contamination_bitwise_equal"] else "不同",
         "全部相同" if all(p["state_tables_equal"].values()) else "不同：" + "、".join(k for k, v in p["state_tables_equal"].items() if not v),
         f'{p["masks_compared"] - len(p["masks_differing"])} / {p["masks_compared"]}', p["box_ok_minutes_on_cut_day_before_cut"],
         f'{"是" if lk["detected_by_state_table_comparison"] else "否"} / '
         f'{"是" if lk["detected_by_levels_only (VAL/POC/VAH/band/primed)"] else "否"}']
        for p, lk in zip(pt, r["planted_leak_check"])]))

    # ---------------- 频次 ----------------
    c = r["counts"]["pct"]
    s.append("\n### 表 7　原始触发分钟数（百分比口径，无箱体）\n")
    rows = []
    for d_, lab in (("up", "Up"), ("dn", "Down")):
        for y in YEARS:
            x = c[d_][y]["raw"]
            rows.append([f"{lab} {YL[y]}", x["only_1"], x["only_2"], x["both"], x["total"], c[d_][y]["onset"]["total"]])
    s.append(table(["方向 / 年份", "仅指标 ①", "仅指标 ②", "① + ② 同时", "合计（分钟）", "独立起点（onset）"], rows))
    s.append("\n### 表 8　叠加日线箱体二段反弹区后的剩余触发（百分比口径，主口径箱体）\n")
    rows = []
    for d_, lab in (("up", "Up"), ("dn", "Down")):
        for y in YEARS:
            x = c[d_][y]["box_raw"]
            tot = c[d_][y]["raw"]["total"]
            rows.append([f"{lab} {YL[y]}", x["only_1"], x["only_2"], x["both"], x["total"],
                         f'{x["total"] / tot * 100:.1f}%' if tot else "—", c[d_][y]["box_onset"]["total"]])
    s.append(table(["方向 / 年份", "仅指标 ①", "仅指标 ②", "① + ② 同时", "合计（分钟）", "占无箱体", "独立起点"], rows))
    s.append("\n### 表 9　按持仓互斥与冷却推算的可成交笔数（只用触发时间戳：持满 2 根 / 只持 1 根 / 含止损上界）\n")
    bi, bs = c["bi"], c["bi_skip_ambiguous"]
    rows = []
    for y in YEARS:
        f3 = lambda x: f'{x["sched45"]["total"]:,} / {x["sched30"]["total"]:,} / {x["sched16"]["total"]:,}'
        rows.append([YL[y], f3({k: c["up"][y]["box_" + k] for k in ("sched45", "sched30", "sched16")}),
                     f3({k: c["dn"][y]["box_" + k] for k in ("sched45", "sched30", "sched16")}),
                     f3(c["up"][y]), f3(bi[y]), f3(bs[y]), f'{c["up"][y]["sched30"]["total"]:,}'])
    s.append(table(["年份", "A1/A2/A3（Up + 箱体）", "A4（Down + 箱体）", "B1/B2（Up）", "B3（方向不明计入）",
                    "B3（方向不明跳过）", "B4（Up，固定 1 根，无提前离场）"], rows))
    sp45 = c["up"]["IS"]["sched45"]
    s.append(f'\n每格三个数依次为 sched45（每笔持满 2 根 15m，下一触发须间隔 ≥ 45 分钟）、sched30（每笔只持 1 根、无提前离场，≥ 30 分钟）'
             f'与 sched16（止损在入场后第 1 根 1m 内离场时的上界，≥ 16 分钟）。B 组 Up 的 sched45 集合中仅 ① / 仅 ② / 同时 = '
             f'{sp45["only_1"]:,} / {sp45["only_2"]:,} / {sp45["both"]:,}；B3 sched45 中方向不明的分钟 {bi["IS"]["sched45"]["ambiguous"]} 个。')
    z = r["counts"]["z"]
    zr = r["z_score_rule"]
    s.append("\n### 表 10　Z-score 稳健性口径（阈值按 σ7d / σ_ref 缩放）\n")
    rows = []
    for d_, lab in (("up", "Up"), ("dn", "Down")):
        for y in YEARS:
            rows.append([f"{lab} {YL[y]}", z[d_][y]["raw"]["total"], c[d_][y]["raw"]["total"],
                         f'{z[d_][y]["sched45"]["total"]:,}–{z[d_][y]["sched30"]["total"]:,}',
                         f'{z[d_][y]["box_sched45"]["total"]:,}–{z[d_][y]["box_sched30"]["total"]:,}'])
    s.append(table(["方向 / 年份", "Z 口径原始分钟", "百分比口径原始分钟", "Z：无箱体可成交", "Z：有箱体可成交"], rows))
    q_ = zr["scale_quantiles"]
    pb = zr["pct_triggers_before_first_sigma7d"]
    s.append(f'\nσ_ref = {zr["sigma_ref_1m"] * 1e4:.2f} bp / 分钟（IS 可交易分钟 σ7d 的中位数，IS 内标定的常数）；阈值缩放倍数 σ7d/σ_ref 的 '
             f'5% / 25% / 75% / 95% 分位为 {q_["0.05"]:.2f} / {q_["0.25"]:.2f} / {q_["0.75"]:.2f} / {q_["0.95"]:.2f}；首个 σ7d 有效分钟 '
             f'{zr["first_minute_with_sigma7d"][:16]}，在此之前百分比口径有 Up {pb["up"]} / Down {pb["dn"]} 个触发分钟（Z 口径无定义）。')
    s.append("\n### 表 11　箱体口径敏感性（只计数，IS 合计；可成交为 sched45–sched30）\n")
    names = {"main": "主口径", "S1a_profile_1h": "S1a 分布用 1H K 线", "S1b_profile_1d": "S1b 分布用 60 根日线",
             "S2_same_day_bounce": "S2 允许触及当日反弹", "S3_origin_ref": "S3 区间判定用脉冲起点收盘价",
             "S4_spot_stitched_warmup": "S4 现货 1m 成交量拼接预热", "S5_point_in_time": "S5 触及按当日支撑带",
             "S6_reentry_above_poc": "S6 离开区间（越过 POC）后再回到", "S7_ref_below_bounce": "S7 当前价须低于反弹线",
             "S8_bounce_close": "S8 反弹按日线收盘价判定", "S9_profile_1d_spot_warmup": "S9 日线分布 + 现货日线预热"}
    rows = []
    for k, lab in names.items():
        x = r["box_sensitivity"][k]
        stt = x["state"]
        base45 = r["box_sensitivity"]["main"]["up"]["IS"]["sched45"]
        rows.append([lab, f'{stt["primed_days"]} / {stt["valid_days"]}' + ("（同主口径）" if stt["uses_main_state"] else ""),
                     f'{x["up"]["IS"]["raw"]:,} / {x["up"]["IS"]["sched45"]:,}–{x["up"]["IS"]["sched30"]:,}',
                     f'{x["dn"]["IS"]["raw"]:,} / {x["dn"]["IS"]["sched45"]:,}–{x["dn"]["IS"]["sched30"]:,}',
                     f'{(x["up"]["IS"]["sched45"] / base45 - 1) * 100:+.0f}%',
                     " / ".join(f'{x["up"][y]["sched45"]}' for y in YEARS[:4])])
    s.append(table(["口径", "primed 日 / 有效日", "Up：原始 / 可成交", "Down：原始 / 可成交", "Up sched45 相对主口径",
                    "Up sched45 分年（2020/21/22/23）"], rows))
    s.append("\n### 表 12　集中度与最小可检测效应（sched45 集合；MDE 为独立同分布近似，双侧 5%、功效 80%）\n")
    rows = []
    for fam, lab in (("A_up (A1/A2/A3)", "A1–A3"), ("A_down (A4)", "A4"), ("B_up (B1/B2/B4)", "B1/B2/B4"), ("B_bi (B3)", "B3")):
        cc = r["concentration"][fam]["sched45"]
        pw = r["power"][fam]
        rows.append([lab, cc["n"], f'{cc["months_with_events"]} / {cc["effective_number_of_months"]:.1f}', cc["days_with_events"],
                     f'{cc["top5_months_share"] * 100:.1f}%', f'{cc["max_events_one_day"]}（{cc["max_day"]}）',
                     f'{pw["sched45_15m"]["mde_bps_sigma7d"]:.1f}–{pw["sched45_15m"]["mde_bps_sigma60"]:.1f}',
                     f'{pw["sched45_30m"]["mde_bps_sigma7d"]:.1f}–{pw["sched45_30m"]["mde_bps_sigma60"]:.1f}'])
    s.append(table(["版本族", "笔数", "有事件的月份 / 有效月份数", "有事件的天数", "前 5 个月占比", "单日最多",
                    "MDE 15m（bp）", "MDE 30m（bp）"], rows))
    top = r["concentration"]["A_up (A1/A2/A3)"]["sched45"]["top5_months"]
    s.append("\nA1–A3 事件最多的 5 个月：" + "、".join(f"{k}（{v}）" for k, v in top.items()) + "。")
    return "\n".join(s) + "\n"


if __name__ == "__main__":
    res = json.loads((CP1 / "cp1_results.json").read_text())
    md = render(res)
    (CP1 / "cp1_tables.md").write_text(md)
    print(md)

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
    return (f"{x:,}" if isinstance(x, int) else str(x)).replace("|", "\\|")


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(n(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render(r: dict) -> str:
    s = []
    dl = r["download"]
    names = {"perp_1m": "永续 1m 月度（主数据）", "perp_15m_official": "永续 15m 月度（仅核验）",
             "perp_1d_official": "永续 1d 月度（仅核验）", "spot_1d_warmup": "现货 1d 2019-09…12（ATR 预热）",
             "spot_1m_warmup": "现货 1m 2019-11…12（仅 S4 敏感性）"}
    s.append("### 表 1　下载与 SHA256 校验\n")
    s.append(table(["数据集", "文件数", "行数", "SHA256 与官方 CHECKSUM 一致", "带表头的文件（自）"],
                   [[names[k], dl[k]["files"], dl[k]["rows"], "全部一致" if dl[k]["all_checksum_verified"] else "否",
                     f'{dl[k]["files_with_header"]}（{dl[k]["first_with_header"]}）' if dl[k]["first_with_header"] else "0"]
                    for k in names]))
    q = r["qa_integrity_1m"]
    s.append("\n### 表 2　1m 数据完整性（IS 2020-01-01 00:00 至 2023-12-31 23:59）\n")
    s.append(table(["检查项", "结果"], [
        ["行数 / 应有分钟数", f'{q["rows"]:,} / {q["expected_rows"]:,}'],
        ["缺失分钟 / 重复时间戳 / 不在分钟网格上的行", f'{q["missing_minutes"]} / {q["duplicates"]} / {q["off_grid_rows"]}'],
        ["OHLC 逻辑错误 / 非正价格 / 负成交量 / 空值", f'{q["bad_ohlc_rows"]} / {q["non_positive_price_rows"]} / '
                                               f'{q["negative_volume_rows"]} / {q["nan_cells"]}'],
        ["原始 close_time ≠ open_time + 59,999 ms 的行", q["close_time_mismatch_rows"]],
        ["主动买量 > 成交量 / 成交均价落在高低价之外", f'{q["taker_buy_gt_volume_rows"]} / {q["vwap_outside_high_low_rows"]}'],
        ["零成交分钟（其中维护占位：O=H=L=C）", f'{q["zero_volume_minutes"]}（{q["placeholder_minutes"]}）'],
        ["零成交分钟分布", "；".join(f"{a[:16]} 至 {b[11:16]}（{c} 分钟）" for a, b, c in q["placeholder_ranges"])],
        ["|1m 对数收益| 最大值", f'{q["max_abs_1m_log_return"] * 100:.2f}%（{q["max_abs_1m_log_return_at"][:16]}）'],
        ["|1m 收益| > 2% 的分钟", q["minutes_abs_1m_return_gt_2pct"]],
        ["来源：月度归档 / 日度归档替换（分钟）", f'{q["rows_by_src"]["0"]:,} / {q["rows_by_src"]["1"]:,}'],
    ]))
    s.append("\n### 表 3　跨周期交叉对齐（1m 按 UTC 边界聚合后逐根比对）\n")
    rows = []
    for key, lab in (("qa_1m_vs_repo_1h", "仓库 1H（IS 行）"), ("qa_1m_vs_official_15m", "官方 15m 归档"),
                     ("qa_1m_vs_official_1d", "官方 1d 归档")):
        x = r[key]
        ohlc = max(x["open_mismatch_bars"], x["high_mismatch_bars"], x["low_mismatch_bars"], x["close_mismatch_bars"])
        nb = len(x["ohlc_mismatch_bars_list"])
        rows.append([lab, x["bars_common"], f'{nb}' + (f'（{x["ohlc_mismatch_bars_list"][0][:10]}）' if nb else ""),
                     x["volume_bars_rel_gt_1e-9"], f'{x["volume_max_rel_diff"] * 100:.2f}%',
                     x["count_unequal_bars"], x.get("reference_close_time_mismatch_rows", "—")])
        assert ohlc == nb or nb == 20
    s.append(table(["参照", "共同 K 线", "OHLC 不一致的 K 线", "成交量相对差 > 1e-9", "成交量最大相对差", "成交笔数不等",
                    "参照 close_time 异常"], rows))
    a = r["download"]["adjudicated_daily_archives"][0]
    s.append("\n### 表 4　2023-11-10 三方裁决证据\n")
    st = a["silent_traded_minutes"]
    cd = a["control_days_background"]
    s.append(table(["证据", "月度 1m", "日度 1m", "逐笔聚合（aggTrades）"], [
        ["全天成交量（BTC）", f'{a["day_volume"]["monthly"]:,.3f}', f'{a["day_volume"]["daily"]:,.3f}',
         f'{a["day_volume"]["aggtrades"]:,.3f}'],
        ["零成交但逐笔有成交的分钟", f'{st["monthly"]}（{st["monthly_range"][0][11:16]}–{st["monthly_range"][1][11:16]}，'
                                 f'逐笔成交 {st["aggtrades_volume_in_monthly_silent_minutes"]:,.3f} BTC）', st["daily"], "—"],
        ["1H 聚合与仓库 1H 的 OHLC 不一致", "、".join(h[11:16] for h in a["hours_mismatch_monthly_vs_1h"]),
         len(a["hours_mismatch_daily_vs_1h"]), 0],
        ["规则 v1（逐分钟完全一致）", "—", "未通过" if not a["rule_v1_strict_pass"] else "通过", "—"],
        ["规则 v2 (a)/(b)/(c)", "违反 (a)(c)" if a["rule_v2"]["monthly_violates_a_or_c"] else "—",
         "/".join("通过" if a["rule_v2"][k] else "否" for k in ("a_daily_1h_ohlc_equal", "b_daily_day_volume_equals_aggtrades",
                                                              "c_daily_no_silent_traded_minutes")), "—"],
    ]))
    s.append("\n对照日（月度与日度逐位相同）K 线与逐笔聚合的背景差异：" + "；".join(
        f'{c["day"]}：OHLC 不一致 {c["kline_vs_aggtrades_ohlc_mismatch_minutes"]} 分钟、成交量不一致 '
        f'{c["kline_vs_aggtrades_volume_mismatch_minutes"]} 分钟、全天成交量相等 = '
        f'{abs(c["day_volume_kline"] - c["day_volume_aggtrades"]) < 1e-6 * c["day_volume_aggtrades"]}'
        for c in cd + r["qa_kline_vs_aggtrades_background"]) + "。")
    s.append("\n### 表 5　1m 与 1H 成交量不一致的 10 个小时（逐笔归档定位）\n")
    hours = r["qa_volume_discrepancies_localized"]["hours"]
    mins = r["qa_volume_discrepancies_localized"]["minutes"]
    rows = []
    for h in hours:
        hm = [m for m in mins if m["minute"][:13] == h["hour"][:13] and (m["extreme_differs"] or m["close_differs"])]
        tag = lambda m: ("H/L" if m["extreme_differs"] else "") + ("、" if m["extreme_differs"] and m["close_differs"] else "") \
            + ("C" if m["close_differs"] else "")
        rows.append([h["hour"][:16], f'{h["volume_1m"]:,.3f}', f'{h["volume_ref_1h"] - h["volume_1m"]:+,.3f}',
                     f'{h["volume_aggtrades"] - h["volume_1m"]:+,.3f}',
                     "；".join(f'{m["minute"][11:16]}（{m["volume_aggtrades_minus_1m"]:+,.3f}，{tag(m)}）' for m in hm) or "无"])
    s.append(table(["小时", "1m 合计（BTC）", "仓库 1H − 1m", "逐笔 − 1m", "K 线价格与逐笔不同的分钟（该分钟逐笔 − 1m；H/L 最高或最低价不同，C 收盘价不同）"],
                   rows))
    d = r["qa_daily_available_time"]
    s.append("\n### 表 6　UTC 日线可用时间\n")
    s.append(table(["检查项", "结果"], [
        ["日线根数 / 首日 / 末日", f'{d["days"]:,} / {d["first_day"][:10]} / {d["last_day"][:10]}'],
        ["开盘时间不在 UTC 00:00 / 日线不足 1,440 分钟 / 日期不连续", f'{d["open_time_not_midnight_utc"]} / '
                                                    f'{d["days_with_lt_1440_minutes"]} / {d["consecutive_days_gap"]}'],
        ["available_time ≠ open_time + 24h", d["available_time_ne_open_plus_24h"]],
        ["最后一根日线的 available_time（≤ IS_END）", f'{d["last_available_time"][:19]}（{d["last_available_time_le_is_end"]}）'],
        ["官方 1d：close_time = open_time + 24h − 1ms 的行", f'{d["official_1d_close_time_eq_open_plus_24h_minus_1ms"]:,} / '
                                                    f'{d["official_1d_rows"]:,}'],
    ]))
    sp = r["qa_spot_warmup"]
    s.append("\n现货预热：1d 共 " + f'{sp["spot_1d_rows"]} 根（{sp["spot_1d_first"][:10]} 至 {sp["spot_1d_last"][:10]}，'
             f'缺口 {sp["spot_1d_gaps"]}），与仓库现货 1H 聚合日线 OHLC 不一致 {len(sp["spot_1d_vs_repo_spot_1h_ohlc_mismatch_days"])} 天；'
             f'1m 共 {sp["spot_1m_rows"]:,} 根，缺失 {sp["spot_1m_missing_minutes"]} 分钟（' +
             "；".join(f"{x[0][:16]} 至 {x[1][11:16]}" for x in sp["spot_1m_missing_ranges"]) + "）。")

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
    s.append("\n### 表 9　按持仓互斥与冷却推算的可成交笔数区间（只用触发时间戳）\n")
    rows = []
    for y in YEARS:
        rows.append([YL[y],
                     f'{c["up"][y]["box_sched45"]["total"]:,}–{c["up"][y]["box_sched30"]["total"]:,}',
                     f'{c["dn"][y]["box_sched45"]["total"]:,}–{c["dn"][y]["box_sched30"]["total"]:,}',
                     f'{c["up"][y]["sched45"]["total"]:,}–{c["up"][y]["sched30"]["total"]:,}',
                     f'{c["bi"][y]["sched45"]["total"]:,}–{c["bi"][y]["sched30"]["total"]:,}',
                     f'{c["up"][y]["sched30"]["total"]:,}'])
    s.append(table(["年份", "A1/A2/A3（Up + 箱体）", "A4（Down + 箱体）", "B1/B2（Up）", "B3（Up 多 + Down 空）",
                    "B4（Up，固定 1 根）"], rows))
    sp45 = c["up"]["IS"]["sched45"]
    s.append(f'\n区间左端为每笔都持满 2 根 15m（下一触发须间隔 ≥ 45 分钟），右端为每笔只持 1 根（≥ 30 分钟）；止损提前离场会让实际笔数超过右端。'
             f'B 组 Up 的 sched45 集合中仅 ① / 仅 ② / 同时 = {sp45["only_1"]:,} / {sp45["only_2"]:,} / {sp45["both"]:,}。')
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
    s.append(f'\nσ_ref = {zr["sigma_ref_1m"] * 1e4:.2f} bp / 分钟（IS 可交易分钟 σ7d 的中位数）；阈值缩放倍数 σ7d/σ_ref 的 5% / 25% / 75% / 95% '
             f'分位为 {q_["0.05"]:.2f} / {q_["0.25"]:.2f} / {q_["0.75"]:.2f} / {q_["0.95"]:.2f}；首个 σ7d 有效分钟 '
             f'{zr["first_minute_with_sigma7d"][:16]}。')
    s.append("\n### 表 11　箱体口径敏感性（只计数，IS 合计：原始分钟 / 可成交区间）\n")
    names = {"main": "主口径", "S1a_profile_1h": "S1a 分布用 1H K 线", "S1b_profile_1d": "S1b 分布用日线",
             "S2_same_day_bounce": "S2 允许触及当日反弹", "S3_origin_ref": "S3 区间判定用脉冲起点收盘价",
             "S4_spot_stitched_warmup": "S4 现货成交量拼接预热", "S5_point_in_time": "S5 触及按当日支撑带"}
    rows = []
    for k, lab in names.items():
        x = r["box_sensitivity"][k]
        stt = x.get("state") or r["box_sensitivity"]["main"]["state"]
        rows.append([lab, f'{stt["primed_days"]} / {stt["valid_days"]}',
                     f'{x["up"]["IS"]["raw"]:,} / {x["up"]["IS"]["sched45"]:,}–{x["up"]["IS"]["sched30"]:,}',
                     f'{x["dn"]["IS"]["raw"]:,} / {x["dn"]["IS"]["sched45"]:,}–{x["dn"]["IS"]["sched30"]:,}',
                     " / ".join(f'{x["up"][y]["sched45"]}' for y in YEARS[:4])])
    s.append(table(["口径", "primed 日 / 有效日", "Up：原始 / 可成交", "Down：原始 / 可成交", "Up sched45 分年（2020/21/22/23）"], rows))
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

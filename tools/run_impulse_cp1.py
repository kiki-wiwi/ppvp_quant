#!/usr/bin/env python3
"""
tools/run_impulse_cp1.py — 脉冲研究 Checkpoint 1：1m 数据工程校验与“盲收益”触发频次摸底。

用法（仓库根目录）：
    python -m impulse_study.data_1m --build      # 首次：下载并校验官方归档，写缓存与 data/manifest_impulse_1m.json
    python tools/run_impulse_cp1.py              # 核验 + 纯频次统计，输出 reports/impulse_vs_box_round/cp1/

范围：BTCUSDT 永续 IS [2020-01-01, 2024-01-01) 与 2020 年之前的现货预热数据。不读取 VAL / OOS / ETH。
本脚本不读取任何触发后的价格、不计算任何收益或盈亏：所有计数只依赖触发时刻及之前的数据与触发时间戳本身。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data_loader import load_klines  # noqa: E402
from impulse_study import daily as DY  # noqa: E402
from impulse_study import data_1m as D  # noqa: E402
from impulse_study import qa as Q  # noqa: E402
from impulse_study import signals as S  # noqa: E402
from impulse_study.scope import IS_END, IS_START, assert_within_is  # noqa: E402
from ppvp_audit.common import load_is_market  # noqa: E402

OUT = ROOT / "reports" / "impulse_vs_box_round" / "cp1"
YEARS = (2020, 2021, 2022, 2023)
VARIANTS = ("main", "S1a_profile_1h", "S1b_profile_1d", "S2_same_day_bounce", "S3_origin_ref",
            "S4_spot_stitched_warmup", "S5_point_in_time")
PREFIX_CUTS = ("2020-03-12 13:37", "2021-05-19 13:37", "2022-11-09 00:00", "2023-12-31 23:59")
CONTROL_DAYS = ("2020-03-12", "2021-05-19")       # 高波动对照日：K 线与逐笔聚合的背景差异


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": run("status", "--porcelain").splitlines()}


def prefix_truncation_check(df1m, s1d, s1m, f_full, conds_full, box_full) -> list[dict]:
    """在若干截断点 τ 上只用 open_time < τ 的 1m 数据重算日线状态、箱体过滤与触发条件，与全量结果逐分钟比对。"""
    out = []
    t_full = np.asarray(f_full.index.as_unit("ns").asi8)
    for cut in PREFIX_CUTS:
        tau = pd.Timestamp(cut, tz="UTC")
        dft = df1m[df1m.index < tau]
        st = DY.build_box_states(dft, s1d, s1m, variants=("main",))["main"]
        g = D.minute_grid(dft, IS_START, tau)
        ft = S.impulse_frame(g)
        n = len(ft)
        ct = S.conditions(ft)
        bt = DY.minute_box_ok(t_full[:n], ft["close"].to_numpy(), st)
        cols = ["r1", "r5", "r15", "sigma7d", "sigma60"]
        a, b = ft[cols].to_numpy(), f_full[cols].to_numpy()[:n]
        feat_equal = bool(np.array_equal(a, b, equal_nan=True))
        cond_equal = all(np.array_equal(ct[k], conds_full[k][:n]) for k in ct)
        out.append({"cut": str(tau), "minutes_compared": n, "features_bitwise_equal": feat_equal,
                    "trigger_conditions_equal": cond_equal, "box_ok_equal": bool(np.array_equal(bt, box_full[:n])),
                    "box_ok_minutes_before_cut": int(bt.sum())})
    return out


def main() -> dict:
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    timing = {}
    res = {"git": git_state(), "scope": {"symbol": "BTCUSDT", "window": [str(IS_START), str(IS_END)],
                                         "post_trigger_prices_read": False, "pnl_computed": False}}

    # ---------------- 1. 数据载入（内容哈希与 manifest 逐一核对） ----------------
    t0 = time.time()
    man = json.loads(D.MANIFEST_PATH.read_text())
    df = D.load_perp_1m()
    o15, o1d = D.load_official("15m"), D.load_official("1d")
    s1d, s1m = D.load_spot_warmup("1d"), D.load_spot_warmup("1m")
    df1h, _, _ = load_is_market()
    df1h = df1h.copy()
    df1h.index = pd.DatetimeIndex(df1h.index).as_unit("ns")
    assert_within_is(df1h.index, "repo 1H")
    sp = load_klines("BTCUSDT", "1h", "spot")
    sp = sp[(sp.index >= pd.Timestamp("2019-09-01", tz="UTC")) & (sp.index < IS_START)].copy()
    sp.index = pd.DatetimeIndex(sp.index).as_unit("ns")
    timing["load_s"] = time.time() - t0

    # ---------------- 2. 数据核验 ----------------
    t0 = time.time()
    files = man["files"]
    res["download"] = {k: {"files": len(v), "all_checksum_verified": all(f["checksum_verified"] for f in v),
                           "rows": int(sum(f["rows"] for f in v)),
                           "files_with_header": int(sum(f["has_header"] for f in v)),
                           "first_with_header": next((f["url"].rsplit("/", 1)[1] for f in v if f["has_header"]), None)}
                       for k, v in files.items()}
    res["download"]["adjudicated_daily_archives"] = [
        {"day": a["day"], "verdict": a["verdict"], "rule_v1_strict_pass": a["rule_v1_strict_pass"], "rule_v2": a["rule_v2"],
         "silent_traded_minutes": a["silent_traded_minutes"], "day_volume": a["day_volume"],
         "hours_mismatch_monthly_vs_1h": a["hours_mismatch_monthly_vs_1h"],
         "hours_mismatch_daily_vs_1h": a["hours_mismatch_daily_vs_1h"],
         "monthly_vs_daily": a["monthly_vs_daily"], "control_days_background": a.get("control_days_background"),
         "daily_sha256": a["daily_1m_archive"]["sha256"], "aggtrades_sha256": a["aggtrades_archive"]["sha256"]}
        for a in man["adjudicated_days"]]
    res["cache_hashes"] = {k: v["content_sha256"] for k, v in man["caches"].items()}
    res["qa_integrity_1m"] = Q.qa_1m_integrity(df)
    res["qa_1m_vs_repo_1h"] = Q.qa_vs_reference(df, df1h, "1h", "data/processed/BTCUSDT_perp_1h.parquet (IS rows only)")
    res["qa_1m_vs_official_15m"] = Q.qa_vs_reference(df, o15, "15m", "official perp 15m monthly archives")
    res["qa_1m_vs_official_1d"] = Q.qa_vs_reference(df, o1d, "1d", "official perp 1d monthly archives")
    res["qa_daily_available_time"] = Q.qa_daily_available(Q.daily_bars(df), o1d)
    res["qa_spot_warmup"] = Q.qa_spot_warmup(s1d, s1m, sp)
    res["qa_volume_discrepancies_localized"] = Q.localize_volume_discrepancies(df, df1h)
    res["qa_kline_vs_aggtrades_background"] = [D._kline_vs_agg_background(pd.Timestamp(d, tz="UTC"), df)
                                               for d in CONTROL_DAYS]
    timing["qa_s"] = time.time() - t0

    # ---------------- 3. 日线状态（主口径与敏感性） ----------------
    t0 = time.time()
    states = DY.build_box_states(df, s1d, s1m, variants=tuple(v for v in VARIANTS if v != "S3_origin_ref"))
    timing["daily_states_s"] = time.time() - t0
    main_st = states["main"]
    valid_days = main_st.index[main_st["valid"].to_numpy()]
    res["daily_state_main"] = {
        "as_of_days": int(len(main_st)), "valid_days": int(main_st["valid"].sum()),
        "first_valid_as_of": str(valid_days.min()), "primed_days": int(main_st["primed"].sum()),
        "atr_first_valid_spot_warmup": str(states["atr"].first_valid_index()),
        "median_band_halfwidth_pct_of_val": float((0.5 * main_st["ATR"] / main_st["VAL"]).median() * 100),
        "median_zone_width_pct_of_val": float(((main_st["POC"] - main_st["band_lo"]) / main_st["VAL"]).median() * 100),
        "median_atr_pct": float((main_st["ATR"] / main_st["VAL"]).median() * 100)}
    st_csv = main_st.copy()
    st_csv.index = st_csv.index.strftime("%Y-%m-%d")
    st_csv.to_csv(OUT / "daily_box_state_main.csv", float_format="%.6f")

    # ---------------- 4. 触发器与纯频次 ----------------
    t0 = time.time()
    grid = D.minute_grid(df)
    f = S.impulse_frame(grid)
    timing["impulse_frame_s"] = time.time() - t0
    tns = np.asarray(f.index.as_unit("ns").asi8)
    yrs = S.years_of(f.index)
    win = S.eligible_window(f.index)
    contam = f["contaminated"].to_numpy()
    base = win & ~contam
    close = f["close"].to_numpy()
    c15 = f["close"].shift(15).to_numpy()
    c5 = f["close"].shift(5).to_numpy()

    def box_mask(variant: str, cond: dict, direction: str) -> np.ndarray:
        if variant == "S3_origin_ref":
            ref = np.where(cond[direction + "2"], c15, c5)
            return DY.minute_box_ok(tns, ref, main_st)
        return DY.minute_box_ok(tns, close, states[variant])

    sig_ref = float(np.nanmedian(f["sigma7d"].to_numpy()[win]))
    scale_z = (f["sigma7d"] / sig_ref).to_numpy()
    rules = {"pct": S.conditions(f), "z": S.conditions(f, scale_z)}
    res["z_score_rule"] = {"sigma_ref_1m": sig_ref, "sigma_ref_definition": "median over IS trigger-eligible minutes of "
                           "sigma7d(T) = RMS of the 10,080 one-minute simple returns ending at T-15",
                           "first_minute_with_sigma7d": str(f["sigma7d"].first_valid_index()),
                           "equivalent_pct_thresholds_at_sigma_ref": dict(S.THR),
                           "scale_quantiles": {str(q): float(np.nanquantile(scale_z[win], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}
    res["eligibility"] = {"first_eligible_minute": str(f.index[win][0]), "last_eligible_minute": str(f.index[win][-1]),
                          "contaminated_minutes": int(contam.sum()),
                          "triggers_excluded_contaminated": {
                              r: {d: int(((c[d + "1"] | c[d + "2"]) & contam & win).sum()) for d in ("up", "dn")}
                              for r, c in rules.items()},
                          "triggers_excluded_outside_window": {
                              r: {d: int(((c[d + "1"] | c[d + "2"]) & ~win).sum()) for d in ("up", "dn")}
                              for r, c in rules.items()},
                          "near_threshold_minutes_detail": [
                              {"minute": str(f.index[i]), "series": k, "value": float(f[k].iloc[i]), "threshold": S.THR[k],
                               "any_trigger_condition": bool(any(rules["pct"][q][i] for q in rules["pct"]))}
                              for k in ("r1", "r5", "r15")
                              for i in np.flatnonzero(np.abs(np.abs(f[k].to_numpy()) - S.THR[k]) < 1e-9)],
                          "near_threshold_minutes_abs_diff_lt_1e-9": int(sum(
                              (np.abs(np.abs(f[k].to_numpy()) - S.THR[k]) < 1e-9).sum() for k in ("r1", "r5", "r15")))}

    counts, sched_sets, boxes = {}, {}, {}
    for r, c in rules.items():
        counts[r] = {}
        for d in ("up", "dn"):
            bx = box_mask("main", c, d)
            boxes[(r, d)] = bx
            tab, sets = S.direction_table(c[d + "1"], c[d + "2"], base, bx, yrs, YEARS)
            counts[r][d] = tab
            sched_sets[(r, d)] = sets
        # B3：涨超做多 + 跌超做空，共用一个持仓
        bi = ((c["up1"] | c["up2"]) | (c["dn1"] | c["dn2"])) & base
        counts[r]["bi"] = {}
        for blk in (30, 45):
            acc = S.schedule(bi, blk)
            sched_sets[(r, "bi")] = sched_sets.get((r, "bi"), {})
            sched_sets[(r, "bi")][f"sched{blk}"] = acc
            for y in list(YEARS) + ["IS"]:
                ys = np.ones_like(acc) if y == "IS" else (yrs == y)
                counts[r]["bi"].setdefault(str(y), {})[f"sched{blk}"] = {
                    "total": int((acc & ys).sum()),
                    "long_up": int((acc & ys & (c["up1"] | c["up2"])).sum()),
                    "short_down": int((acc & ys & (c["dn1"] | c["dn2"])).sum())}
    res["counts"] = counts
    # 同一分钟同时满足向上与向下条件（例如 1m + 5m 急涨叠加 15m 累计急跌）
    res["up_down_conflicts"] = {}
    for r, c in rules.items():
        upm, dnm = (c["up1"] | c["up2"]) & base, (c["dn1"] | c["dn2"]) & base
        both = upm & dnm
        res["up_down_conflicts"][r] = {
            "minutes": int(both.sum()), "up1_and_dn2": int((c["up1"] & c["dn2"] & base).sum()),
            "dn1_and_up2": int((c["dn1"] & c["up2"] & base).sum()),
            "by_year": {str(y): int((both & (yrs == y)).sum()) for y in YEARS},
            "accepted_in_B3_sched45": int((both & sched_sets[(r, "bi")]["sched45"]).sum()),
            "accepted_in_B3_sched30": int((both & sched_sets[(r, "bi")]["sched30"]).sum())}
    res["daily_state_main"]["box_ok_minute_share_all_IS_minutes"] = float(DY.minute_box_ok(tns, close, main_st).mean())
    timing["counts_s"] = time.time() - t0

    # 箱体口径敏感性（只计数）
    sens = {}
    c = rules["pct"]
    for v in VARIANTS:
        sens[v] = {}
        for d in ("up", "dn"):
            bx = box_mask(v, c, d)
            trig = (c[d + "1"] | c[d + "2"]) & base & bx
            on = S.onset(c[d + "1"] | c[d + "2"]) & base & bx
            a45, a30 = S.schedule(trig, 45), S.schedule(trig, 30)
            sens[v][d] = {str(y): {"raw": int((trig & (yrs == y if y != "IS" else True)).sum()),
                                   "onset": int((on & (yrs == y if y != "IS" else True)).sum()),
                                   "sched45": int((a45 & (yrs == y if y != "IS" else True)).sum()),
                                   "sched30": int((a30 & (yrs == y if y != "IS" else True)).sum())}
                          for y in list(YEARS) + ["IS"]}
        if v != "S3_origin_ref":
            stv = states[v]
            sens[v]["state"] = {"valid_days": int(stv["valid"].sum()), "primed_days": int(stv["primed"].sum()),
                                "first_valid_as_of": str(stv.index[stv["valid"].to_numpy()].min())}
    res["box_sensitivity"] = sens

    # 集中度、时段、功效（只用触发时刻及之前的波动率）
    sig7, sig60 = f["sigma7d"].to_numpy(), f["sigma60"].to_numpy()
    families = {"B_up (B1/B2/B4)": (("pct", "up"), "sched45", "sched30"),
                "B_bi (B3)": (("pct", "bi"), "sched45", "sched30"),
                "A_up (A1/A2/A3)": (("pct", "up"), "box_sched45", "box_sched30"),
                "A_down (A4)": (("pct", "dn"), "box_sched45", "box_sched30")}
    conc, power = {}, {}
    for name, (key, k45, k30) in families.items():
        a45, a30 = sched_sets[key][k45], sched_sets[key][k30]
        conc[name] = {"sched45": S.concentration(a45, f.index), "sched30": S.concentration(a30, f.index)}
        power[name] = {f"{k}_{h}m": {"n": int(acc.sum()),
                                     "mde_bps_sigma7d": S.mde_bps(sig7[acc], h),
                                     "mde_bps_sigma60": S.mde_bps(sig60[acc], h)}
                       for k, acc in (("sched45", a45), ("sched30", a30)) for h in (15, 30)}
        power[name]["median_sigma7d_bps"] = float(np.nanmedian(sig7[a45]) * 1e4)
        power[name]["median_sigma60_bps"] = float(np.nanmedian(sig60[a45]) * 1e4)
    res["concentration"] = conc
    res["power"] = power

    # 月度、时段分布（CSV）
    mon = pd.DataFrame(index=pd.period_range("2020-01", "2023-12", freq="M").astype(str))
    hr = pd.DataFrame(index=range(24))
    ym = pd.DatetimeIndex(f.index).strftime("%Y-%m").to_numpy()
    hh = pd.DatetimeIndex(f.index).hour.to_numpy()
    for (r, d), sets in sched_sets.items():
        if r != "pct":
            continue
        for k, acc in sets.items():
            mon[f"{d}_{k}"] = pd.Series(ym[acc]).value_counts().reindex(mon.index).fillna(0).astype(int)
            hr[f"{d}_{k}"] = pd.Series(hh[acc]).value_counts().reindex(hr.index).fillna(0).astype(int)
    for d in ("up", "dn"):
        cc = rules["pct"]
        trig = (cc[d + "1"] | cc[d + "2"]) & base
        mon[f"{d}_raw"] = pd.Series(ym[trig]).value_counts().reindex(mon.index).fillna(0).astype(int)
        mon[f"{d}_box_raw"] = pd.Series(ym[trig & boxes[("pct", d)]]).value_counts().reindex(mon.index).fillna(0).astype(int)
    mon.index.name = "month"
    hr.index.name = "utc_hour"
    mon.to_csv(OUT / "monthly_trigger_counts.csv")
    hr.to_csv(OUT / "hourly_trigger_counts.csv")

    # 触发分钟台账（只含触发时刻及之前的信息）
    cc = rules["pct"]
    anyt = ((cc["up1"] | cc["up2"]) | (cc["dn1"] | cc["dn2"])) & win
    led = pd.DataFrame({
        "open_time": f.index[anyt], "direction": np.where((cc["up1"] | cc["up2"])[anyt], "up", "down"),
        "ind1": np.where((cc["up1"] | cc["up2"])[anyt], cc["up1"][anyt], cc["dn1"][anyt]),
        "ind2": np.where((cc["up1"] | cc["up2"])[anyt], cc["up2"][anyt], cc["dn2"][anyt]),
        "close": close[anyt], "r1": f["r1"].to_numpy()[anyt], "r5": f["r5"].to_numpy()[anyt],
        "r15": f["r15"].to_numpy()[anyt], "sigma7d": sig7[anyt], "contaminated": contam[anyt],
        "onset": np.where((cc["up1"] | cc["up2"])[anyt], S.onset(cc["up1"] | cc["up2"])[anyt],
                          S.onset(cc["dn1"] | cc["dn2"])[anyt]),
        "box_main": np.where((cc["up1"] | cc["up2"])[anyt], boxes[("pct", "up")][anyt], boxes[("pct", "dn")][anyt]),
        "z_rule_also": np.where((cc["up1"] | cc["up2"])[anyt], (rules["z"]["up1"] | rules["z"]["up2"])[anyt],
                                (rules["z"]["dn1"] | rules["z"]["dn2"])[anyt]),
        "B_sched45": np.where((cc["up1"] | cc["up2"])[anyt], sched_sets[("pct", "up")]["sched45"][anyt],
                              sched_sets[("pct", "dn")]["sched45"][anyt]),
        "A_sched45": np.where((cc["up1"] | cc["up2"])[anyt], sched_sets[("pct", "up")]["box_sched45"][anyt],
                              sched_sets[("pct", "dn")]["box_sched45"][anyt]),
        "B3_sched45": sched_sets[("pct", "bi")]["sched45"][anyt]})
    led["open_time"] = led["open_time"].dt.strftime("%Y-%m-%d %H:%M")
    led.to_csv(OUT / "trigger_minutes.csv.gz", index=False, float_format="%.8g", compression={"method": "gzip", "mtime": 0})
    res["trigger_ledger"] = {"rows": int(len(led)), "path": str((OUT / "trigger_minutes.csv.gz").relative_to(ROOT))}

    # ---------------- 5. 前缀截断（无未来函数）核验 ----------------
    t0 = time.time()
    res["prefix_truncation"] = prefix_truncation_check(df, s1d, s1m, f, rules["pct"], boxes[("pct", "up")])
    timing["prefix_truncation_s"] = time.time() - t0
    timing["total_s"] = time.time() - t_start
    res["timing"] = timing
    (OUT / "cp1_results.json").write_text(json.dumps(jsonable(res), ensure_ascii=False, indent=1))
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps(jsonable({k: r[k] for k in ("timing", "prefix_truncation", "eligibility")}), indent=1))

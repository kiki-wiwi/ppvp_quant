#!/usr/bin/env python3
"""
tools/run_impulse_cp1.py — 脉冲研究 Checkpoint 1：1m 数据工程校验与“盲收益”触发频次摸底。

用法（仓库根目录）：
    python -m impulse_study.data_1m --rebuild    # 按 manifest（SHA256 锁定）复现缓存；本地有原始 zip 时可加 --offline
    python tools/run_impulse_cp1.py              # 核验 + 纯频次统计，输出 reports/impulse_vs_box_round/cp1/
    python tools/render_impulse_cp1_tables.py    # 由结果 JSON 生成阶段记录中的表格

范围：BTCUSDT 永续 IS [2020-01-01, 2024-01-01) 与 2020 年之前的现货预热数据。不读取 VAL / OOS / ETH：
仓库的 1H 与现货 1H parquet 覆盖到 2026 年，这里以谓词过滤读取，IS 之后的行不返回给程序；不读取资金费文件。
本脚本不读取任何触发后的价格、不计算任何收益或盈亏：所有计数只依赖触发时刻及之前的数据与触发时间戳本身。
QA 部分会下载若干官方日度归档（日度 1m、逐笔聚合成交），其 URL 与 SHA256 写入结果 JSON。
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

from impulse_study import daily as DY  # noqa: E402
from impulse_study import data_1m as D  # noqa: E402
from impulse_study import qa as Q  # noqa: E402
from impulse_study import signals as S  # noqa: E402
from impulse_study.scope import DAY, IS_END, IS_START  # noqa: E402

OUT = ROOT / "reports" / "impulse_vs_box_round" / "cp1"
YEARS = (2020, 2021, 2022, 2023)
VARIANTS = ("main", "S1a_profile_1h", "S1b_profile_1d", "S2_same_day_bounce", "S3_origin_ref",
            "S4_spot_stitched_warmup", "S5_point_in_time", "S6_reentry_above_poc", "S7_ref_below_bounce",
            "S8_bounce_close", "S9_profile_1d_spot_warmup")
STATE_VARIANTS = tuple(v for v in VARIANTS if v not in ("S3_origin_ref", "S7_ref_below_bounce"))
FIXED_CUTS = ("2020-03-12 13:37", "2021-05-19 13:37", "2022-11-09 00:00", "2023-12-31 23:59")
BACKGROUND_DAYS = ("2020-03-12", "2021-05-19")       # 高波动对照日：K 线与逐笔聚合的背景差异


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        if np.isinf(x):
            return "inf" if x > 0 else "-inf"
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


def box_mask(variant: str, cond: dict, direction: str, tns, close, c15, c5, states) -> np.ndarray:
    if variant == "S3_origin_ref":
        ref = np.where(cond[direction + "2"], c15, c5)
        return DY.minute_box_ok(tns, ref, states["main"])
    if variant == "S7_ref_below_bounce":
        return DY.minute_box_ok(tns, close, states["main"], below_bounce=True)
    return DY.minute_box_ok(tns, close, states[variant])


def masks_for(f: pd.DataFrame, states: dict, sig_ref: float) -> dict:
    """一组数据上的全部逐分钟掩码（特征、触发、箱体、起点、日程），供全量与截断结果逐位比对。"""
    tns = np.asarray(f.index.as_unit("ns").asi8)
    close = f["close"].to_numpy()
    c15, c5 = f["close"].shift(15).to_numpy(), f["close"].shift(5).to_numpy()
    base = S.eligible_window(f.index) & ~f["contaminated"].to_numpy()
    out = {"base": base}
    rules = {"pct": S.conditions(f), "z": S.conditions(f, (f["sigma7d"] / sig_ref).to_numpy())}
    for r, c in rules.items():
        for k, v in c.items():
            out[f"{r}_{k}"] = v
    c = rules["pct"]
    for d in ("up", "dn"):
        trig = (c[d + "1"] | c[d + "2"]) & base
        out[f"onset_{d}"] = S.onset(c[d + "1"] | c[d + "2"]) & base
        for blk in S.BLOCK.values():
            out[f"sched{blk}_{d}"] = S.schedule(trig, blk)
        for v in VARIANTS:
            bx = box_mask(v, c, d, tns, close, c15, c5, states)
            out[f"box_{v}_{d}"] = bx
            out[f"box_onset_{v}_{d}"] = S.onset((c[d + "1"] | c[d + "2"]) & bx) & base
            for blk in S.BLOCK.values():
                out[f"box_sched{blk}_{v}_{d}"] = S.schedule(trig & bx, blk)
    return out


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if not a.index.equals(b.index) or list(a.columns) != list(b.columns):
        return False
    return all(np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=a[c].dtype.kind == "f") for c in a.columns)


def select_cuts(f: pd.DataFrame, main_st: pd.DataFrame) -> list[str]:
    """固定截断点之外，每年再取一个“截断日当天截断前已有 ≥ 30 个箱体内分钟”的截断点（规则写死：该年第一个满足条件的
    primed 日的 13:37），使截断核验能检出“当日数据泄漏进当日状态”一类的错误。"""
    tns = np.asarray(f.index.as_unit("ns").asi8)
    bx = DY.minute_box_ok(tns, f["close"].to_numpy(), main_st)
    s = pd.Series(bx, index=f.index)
    before = s[(s.index.hour < 13) | ((s.index.hour == 13) & (s.index.minute < 37))]
    per_day = before.groupby(before.index.floor("D")).sum()
    cuts = []
    for y in YEARS:
        days = per_day[(per_day.index.year == y) & (per_day >= 30)]
        if len(days):
            cuts.append(f"{days.index[0]:%Y-%m-%d} 13:37")
    return list(FIXED_CUTS) + cuts


def prefix_truncation_check(df1m, s1d, s1m, f_full, states_full, sig_ref, cuts) -> list[dict]:
    """在截断点 τ 上只用 open_time < τ 的 1m 数据重算：日线状态（主口径与全部敏感性）逐日比对 as_of ≤ floor(τ) 的整表，
    逐分钟比对 τ 之前的特征、维护污染、两种口径的触发条件、全部箱体掩码、起点与三种日程掩码。σ_ref 取全量值（常数）。"""
    m_full = masks_for(f_full, states_full, sig_ref)
    out = []
    for cut in cuts:
        tau = pd.Timestamp(cut, tz="UTC")
        dft = df1m[df1m.index < tau]
        st = DY.build_box_states(dft, s1d, s1m, variants=STATE_VARIANTS)
        st["S3_origin_ref"] = st["main"]
        ft = S.impulse_frame(D.minute_grid(dft, IS_START, tau))
        n = len(ft)
        cols = ["r1", "r5", "r15", "sigma7d", "sigma60", "contaminated"]
        feat_eq = bool(np.array_equal(ft[cols].to_numpy(dtype=float), f_full[cols].to_numpy(dtype=float)[:n], equal_nan=True))
        state_eq = {}
        for v in STATE_VARIANTS:
            a = st[v]
            b = states_full[v].loc[a.index]
            state_eq[v] = _frames_equal(a, b)
        mt = masks_for(ft, st, sig_ref)
        mask_bad = [k for k in mt if not np.array_equal(mt[k], m_full[k][:n])]
        day0 = tau.floor("D")
        on_day = (ft.index >= day0)
        out.append({"cut": str(tau), "minutes_compared": n, "features_and_contamination_bitwise_equal": feat_eq,
                    "state_tables_equal": state_eq, "masks_compared": len(mt), "masks_differing": mask_bad,
                    "box_ok_minutes_on_cut_day_before_cut": int(mt["box_main_up"][on_day].sum()),
                    "primed_on_cut_day": bool(states_full["main"].loc[day0, "primed"]) if day0 in states_full["main"].index
                    else None})
    return out


def planted_leak_check(df1m, s1d, cut: str) -> dict:
    """植入泄漏：把成交量分布窗口改为 [D − 60d, D + 1d)（含当日），用全量与截断数据各算一次，看截断核验能否检出。"""
    tau = pd.Timestamp(cut, tz="UTC")

    def leaky(df):
        daily = DY.daily_from_1m(df)
        atr = DY.atr_1d(daily, s1d)
        as_of = pd.DatetimeIndex(daily.index).append(pd.DatetimeIndex([daily.index[-1] + DAY]))
        as_of = as_of[as_of <= pd.DatetimeIndex(df.index)[-1].floor("D")]
        prof = DY.rolling_profiles(df, as_of + DAY, window_days=DY.W_PROFILE_DAYS + 1,
                                   min_first_bar=pd.DatetimeIndex(df.index)[0])
        prof.index = as_of
        return DY.box_state(daily, prof, atr)

    full, trunc = leaky(df1m), leaky(df1m[df1m.index < tau])
    b = full.loc[trunc.index]
    diff = {}
    for c in trunc.columns:
        x, y = trunc[c].to_numpy(), b[c].to_numpy()
        ne = ~((x == y) | (pd.isna(x) & pd.isna(y)))
        if ne.any():
            diff[c] = [str(t.date()) for t in trunc.index[ne]][:3]
    return {"cut": str(tau), "leak": "profile window [D-60d, D+1d) includes the as_of day itself",
            "detected_by_state_table_comparison": bool(diff), "differing_columns": diff,
            "detected_by_levels_only (VAL/POC/VAH/band/primed)": bool(set(diff) & {"VAL", "POC", "VAH", "band_lo",
                                                                                 "band_hi", "bounce_level", "primed"})}


def main() -> dict:
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    timing = {}
    res = {"git": git_state(), "scope": {"symbol": "BTCUSDT", "window": [str(IS_START), str(IS_END)],
                                         "post_trigger_prices_read": False, "pnl_computed": False,
                                         "funding_file_read": False,
                                         "repo_parquets_read_with_predicate": ["open_time < 2024-01-01 (perp 1H)",
                                                                               "2019-09-01 <= open_time < 2020-01-01 (spot 1H)"]}}

    # ---------------- 1. 数据载入（内容哈希与 manifest 逐一核对） ----------------
    t0 = time.time()
    man = json.loads(D.MANIFEST_PATH.read_text())
    df = D.load_perp_1m()
    o15, o1d = D.load_official("15m"), D.load_official("1d")
    s1d, s1m = D.load_spot_warmup("1d"), D.load_spot_warmup("1m")
    df1h = D.load_repo_1h_is()
    sp = D.load_repo_spot_1h_warmup()
    timing["load_s"] = time.time() - t0

    # ---------------- 2. 数据核验 ----------------
    t0 = time.time()
    files = man["files"]
    res["download"] = {k: {"files": len(v), "all_checksum_verified": all(f["checksum_verified"] for f in v),
                           "rows": int(sum(f["rows"] for f in v)),
                           "files_with_header": int(sum(f["has_header"] for f in v)),
                           "first_with_header": next((f["url"].rsplit("/", 1)[1] for f in v if f["has_header"]), None)}
                       for k, v in files.items()}
    res["download"]["adjudicated_daily_archives"] = man["adjudicated_days"]
    res["cache_hashes"] = {k: v["content_sha256"] for k, v in man["caches"].items()}
    res["qa_integrity_1m"] = Q.qa_1m_integrity(df)
    res["qa_1m_vs_repo_1h"] = Q.qa_vs_reference(df, df1h, "1h", "data/processed/BTCUSDT_perp_1h.parquet (IS rows only)")
    res["qa_1m_vs_official_15m"] = Q.qa_vs_reference(df, o15, "15m", "official perp 15m monthly archives")
    res["qa_1m_vs_official_1d"] = Q.qa_vs_reference(df, o1d, "1d", "official perp 1d monthly archives")
    res["qa_daily_available_time"] = Q.qa_daily_available(df, Q.daily_bars(df), o1d)
    res["qa_spot_warmup"] = Q.qa_spot_warmup(s1d, s1m, sp)
    loc = Q.localize_volume_discrepancies(df, df1h, o1d)
    res["qa_volume_discrepancies_localized"] = loc
    loc_days = [pd.Timestamp(d["day"], tz="UTC") for d in loc["days"]]
    adj_days = [pd.Timestamp(a["day"], tz="UTC") for a in man["adjudicated_days"]]
    ctrl_days = [pd.Timestamp(c["day"], tz="UTC") for a in man["adjudicated_days"] for c in a.get("control_days_background", [])]
    res["qa_monthly_vs_daily_archive"] = Q.monthly_vs_daily_check(df, sorted(set(loc_days + adj_days + ctrl_days)))
    res["qa_maintenance_windows_vs_aggtrades"] = Q.maintenance_windows_vs_aggtrades(df)
    res["qa_kline_vs_aggtrades_background"] = [Q.background_day(pd.Timestamp(d, tz="UTC"), df) for d in BACKGROUND_DAYS]
    timing["qa_s"] = time.time() - t0

    # ---------------- 3. 日线状态（主口径与敏感性） ----------------
    t0 = time.time()
    states = DY.build_box_states(df, s1d, s1m, variants=STATE_VARIANTS)
    states["S3_origin_ref"] = states["S7_ref_below_bounce"] = states["main"]
    timing["daily_states_s"] = time.time() - t0
    main_st = states["main"]
    valid_days = main_st.index[main_st["valid"].to_numpy()]
    vs = main_st[main_st["valid"]]
    res["daily_state_main"] = {
        "as_of_days": int(len(main_st)), "valid_days": int(main_st["valid"].sum()),
        "first_valid_as_of": str(valid_days.min()), "primed_days": int(main_st["primed"].sum()),
        "atr_first_valid_spot_warmup": str(states["atr"].first_valid_index()),
        "median_band_halfwidth_pct_of_val": float((0.5 * vs["ATR"] / vs["VAL"]).median() * 100),
        "median_zone_width_pct_of_val": float(((vs["POC"] - vs["band_lo"]) / vs["VAL"]).median() * 100),
        "median_atr_pct": float((vs["ATR"] / vs["VAL"]).median() * 100),
        "median_poc_minus_val_in_atr": float(((vs["POC"] - vs["VAL"]) / vs["ATR"]).median()),
        "share_valid_days_poc_above_bounce_level": float((vs["POC"] > vs["bounce_level"]).mean()),
        "primed_days_poc_above_bounce_level": int((vs["primed"] & (vs["POC"] > vs["bounce_level"])).sum())}
    st_csv = main_st.copy()
    st_csv.index = st_csv.index.strftime("%Y-%m-%d")
    st_csv.to_csv(OUT / "daily_box_state_main.csv", float_format="%.6f")

    # ---------------- 4. 触发器与纯频次 ----------------
    t0 = time.time()
    f = S.impulse_frame(D.minute_grid(df))
    timing["impulse_frame_s"] = time.time() - t0
    t0 = time.time()
    tns = np.asarray(f.index.as_unit("ns").asi8)
    yrs = S.years_of(f.index)
    win = S.eligible_window(f.index)
    contam = f["contaminated"].to_numpy()
    base = win & ~contam
    close = f["close"].to_numpy()
    c15, c5 = f["close"].shift(15).to_numpy(), f["close"].shift(5).to_numpy()
    sig_ref = float(np.nanmedian(f["sigma7d"].to_numpy()[win]))
    scale_z = (f["sigma7d"] / sig_ref).to_numpy()
    rules = {"pct": S.conditions(f), "z": S.conditions(f, scale_z)}
    first_z = f["sigma7d"].first_valid_index()
    res["z_score_rule"] = {"sigma_ref_1m": sig_ref, "sigma_ref_definition": "median over IS trigger-eligible minutes of "
                           "sigma7d(T) = RMS of the 10,080 one-minute simple returns ending at T-15; an IS-calibrated "
                           "constant (not causal within IS), to be frozen for any later use",
                           "first_minute_with_sigma7d": str(first_z),
                           "pct_triggers_before_first_sigma7d": {
                               d: int(((rules["pct"][d + "1"] | rules["pct"][d + "2"]) & base & (f.index < first_z)).sum())
                               for d in ("up", "dn")},
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
            bx = box_mask("main", c, d, tns, close, c15, c5, states)
            boxes[(r, d)] = bx
            tab, sets = S.direction_table(c[d + "1"], c[d + "2"], base, bx, yrs, YEARS)
            counts[r][d] = tab
            sched_sets[(r, d)] = sets
        # B3：涨超做多 + 跌超做空，共用一个持仓。两种处理：方向不明的分钟计入（两边都记）/ 跳过
        upm, dnm = (c["up1"] | c["up2"]) & base, (c["dn1"] | c["dn2"]) & base
        for key, m in (("bi", upm | dnm), ("bi_skip_ambiguous", upm ^ dnm)):
            counts[r][key] = {}
            sched_sets[(r, key)] = {}
            for name, blk in S.BLOCK.items():
                acc = S.schedule(m, blk)
                sched_sets[(r, key)][name] = acc
                for y in list(YEARS) + ["IS"]:
                    ys = np.ones_like(acc) if y == "IS" else (yrs == y)
                    counts[r][key].setdefault(str(y), {})[name] = {
                        "total": int((acc & ys).sum()), "long_up": int((acc & ys & upm).sum()),
                        "short_down": int((acc & ys & dnm).sum()), "ambiguous": int((acc & ys & upm & dnm).sum())}
    res["counts"] = counts
    res["up_down_conflicts"] = {}
    for r, c in rules.items():
        upm, dnm = (c["up1"] | c["up2"]) & base, (c["dn1"] | c["dn2"]) & base
        both = upm & dnm
        res["up_down_conflicts"][r] = {
            "minutes": int(both.sum()), "up1_and_dn2": int((c["up1"] & c["dn2"] & base).sum()),
            "dn1_and_up2": int((c["dn1"] & c["up2"] & base).sum()),
            "by_year": {str(y): int((both & (yrs == y)).sum()) for y in YEARS},
            **{f"accepted_in_B3_{k}": int((both & v).sum()) for k, v in sched_sets[(r, "bi")].items()},
            **{f"accepted_in_up_{k}": int((both & v).sum()) for k, v in sched_sets[(r, "up")].items() if not k.startswith("box")},
            **{f"accepted_in_dn_{k}": int((both & v).sum()) for k, v in sched_sets[(r, "dn")].items() if not k.startswith("box")}}
    res["daily_state_main"]["box_ok_minute_share_all_IS_minutes"] = float(boxes[("pct", "up")].mean())
    timing["counts_s"] = time.time() - t0

    # 箱体口径敏感性（只计数）
    sens = {}
    c = rules["pct"]
    for v in VARIANTS:
        sens[v] = {}
        for d in ("up", "dn"):
            bx = box_mask(v, c, d, tns, close, c15, c5, states)
            trig = (c[d + "1"] | c[d + "2"]) & base & bx
            on = S.onset((c[d + "1"] | c[d + "2"]) & bx) & base
            acc = {name: S.schedule(trig, blk) for name, blk in S.BLOCK.items()}
            sens[v][d] = {str(y): {"raw": int((trig & (yrs == y if y != "IS" else True)).sum()),
                                   "onset": int((on & (yrs == y if y != "IS" else True)).sum()),
                                   **{name: int((a & (yrs == y if y != "IS" else True)).sum()) for name, a in acc.items()}}
                          for y in list(YEARS) + ["IS"]}
        stv = states[v]
        sens[v]["state"] = {"valid_days": int(stv["valid"].sum()), "primed_days": int(stv["primed"].sum()),
                            "first_valid_as_of": str(stv.index[stv["valid"].to_numpy()].min()),
                            "uses_main_state": v in ("S3_origin_ref", "S7_ref_below_bounce")}
    res["box_sensitivity"] = sens

    # 集中度、时段、功效（只用触发时刻及之前的波动率）
    sig7, sig60 = f["sigma7d"].to_numpy(), f["sigma60"].to_numpy()
    families = {"B_up (B1/B2/B4)": (("pct", "up"), ""), "B_bi (B3)": (("pct", "bi"), ""),
                "A_up (A1/A2/A3)": (("pct", "up"), "box_"), "A_down (A4)": (("pct", "dn"), "box_")}
    conc, power = {}, {}
    for name, (key, pre) in families.items():
        a45, a30 = sched_sets[key][pre + "sched45"], sched_sets[key][pre + "sched30"]
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

    # 触发分钟台账：每个 (分钟, 方向) 一行，只含触发时刻及之前的信息；同时满足两个方向的分钟出现两行（ambiguous = True）
    cc, cz = rules["pct"], rules["z"]
    upm_all, dnm_all = (cc["up1"] | cc["up2"]) & win, (cc["dn1"] | cc["dn2"]) & win
    rows = []
    for d, m, dname in (("up", upm_all, "up"), ("dn", dnm_all, "down")):
        ss = sched_sets[("pct", d)]
        rows.append(pd.DataFrame({
            "open_time": f.index[m], "direction": dname, "ind1": cc[d + "1"][m], "ind2": cc[d + "2"][m],
            "ambiguous_up_and_down": (upm_all & dnm_all)[m], "close": close[m], "r1": f["r1"].to_numpy()[m],
            "r5": f["r5"].to_numpy()[m], "r15": f["r15"].to_numpy()[m], "sigma7d": sig7[m], "contaminated": contam[m],
            "onset": S.onset(cc[d + "1"] | cc[d + "2"])[m], "box_main": boxes[("pct", d)][m],
            "box_onset": S.onset((cc[d + "1"] | cc[d + "2"]) & boxes[("pct", d)])[m],
            "z_rule_also": (cz[d + "1"] | cz[d + "2"])[m],
            **{f"B_{k}": ss[k][m] for k in ("sched45", "sched30", "sched16")},
            **{f"A_{k}": ss["box_" + k][m] for k in ("sched45", "sched30", "sched16")},
            "B3_sched45": sched_sets[("pct", "bi")]["sched45"][m],
            "B3_skip_ambiguous_sched45": sched_sets[("pct", "bi_skip_ambiguous")]["sched45"][m]}))
    led = pd.concat(rows).sort_values(["open_time", "direction"], kind="stable")
    led["open_time"] = led["open_time"].dt.strftime("%Y-%m-%d %H:%M")
    led.to_csv(OUT / "trigger_minutes.csv.gz", index=False, float_format="%.8g", compression={"method": "gzip", "mtime": 0})
    res["trigger_ledger"] = {"rows": int(len(led)), "rows_up": int(upm_all.sum()), "rows_down": int(dnm_all.sum()),
                             "ambiguous_minutes_two_rows": int((upm_all & dnm_all).sum()),
                             "path": str((OUT / "trigger_minutes.csv.gz").relative_to(ROOT))}

    # ---------------- 5. 前缀截断（无未来函数）核验与植入泄漏 ----------------
    t0 = time.time()
    cuts = select_cuts(f, main_st)
    res["prefix_truncation"] = prefix_truncation_check(df, s1d, s1m, f, states, sig_ref, cuts)
    res["planted_leak_check"] = [planted_leak_check(df, s1d, c) for c in cuts]
    timing["prefix_truncation_s"] = time.time() - t0
    timing["total_s"] = time.time() - t_start
    res["timing"] = timing
    (OUT / "cp1_results.json").write_text(json.dumps(jsonable(res), ensure_ascii=False, indent=1))
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps(jsonable({k: r[k] for k in ("timing", "prefix_truncation", "planted_leak_check", "eligibility")}),
                     indent=1)[:8000])

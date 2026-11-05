"""
ppvp_audit/run_ledgers.py — 生成机会级 / 成交级账本、核对表、切片、最差月份核查表、止损后走势与低 RR 分解（IS only）。

用法：python -m ppvp_audit.run_ledgers [--version B_original|B_audited]
输出目录：reports/audit_round/（CSV 与 ledgers_summary_<version>.json）、reports/audit_round/figures/slices/
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import BASELINE_BACKTEST, BASELINE_STRATEGY
from metrics import daily_equity, monthly_returns

from . import ledgers as L
from . import slices as S
from .common import FIG, OUT, is_run, write_json


def build(version: str, scfg=BASELINE_STRATEGY, bcfg=BASELINE_BACKTEST, out: dict | None = None, plots: bool = True) -> dict:
    out = out if out is not None else is_run(scfg, bcfg)
    op = L.opportunities(out, scfg, bcfg, version)
    tl = L.trade_ledger(out, scfg, bcfg, version)
    add = op[op["status"] == "filled"].set_index("signal_bar")
    keep = ["box_key", "first_test_start_bar", "first_test_start_time", "profile_id_at_first_test_start",
            "chain_start_bar", "chain_start_time", "profile_id_at_chain_start", "n_rolls", "profile_id_at_t1_low",
            "profile_id_at_t2_start", "n_profile_switches_since_first_test_start", "n_profile_switches_since_chain_start",
            "profile_changed_since_first_test_start", "profile_changed_since_chain_start",
            "signal_at_profile_supersede_instant", "departure_bar", "departure_height_atr", "second_test_qualified_by",
            "touch_gap_bars_from_t1_last", "confirmation_range_atr", "normalized_entry_position", "next_open_vs_close_bp",
            "pivot_a_time", "pivot_b_time", "profile_available_time", "profile_first_decision_time",
            "profile_superseded_at", "profile_invalidated_at", "profile_window_n_missing", "profile_window_n_imputed",
            "profile_window_n_partial"] + [c for c in op.columns if c.startswith("regime_")]
    tl = tl.join(add[keep], on="entry_bar")
    tl["above_sma200"] = tl["regime_above_sma200"]
    rc = L.reconciliation(tl)
    sfx = "" if version == "B_audited" else f"_{version}"
    bl = L.box_lifecycle(out["boxes"], out["df1h"], out["signals"])
    bl.to_csv(OUT / f"box_lifecycle{sfx}.csv", index=True, index_label="profile_id", float_format="%.10g")
    op.to_csv(OUT / f"signal_opportunities{sfx}.csv", index=False, float_format="%.10g")
    tl.to_csv(OUT / f"trades{sfx}.csv", index=False, float_format="%.10g")
    rc.to_csv(OUT / f"trade_reconciliation{sfx}.csv", index=False, float_format="%.10g")

    # 切片（固定种子）与最差月份 13 笔
    sel = S.select_slices(tl)
    sel.to_csv(OUT / f"slices_selected{sfx}.csv", index=False, float_format="%.10g")
    res = out["result"]
    d = daily_equity(res)
    mr = monthly_returns(d).stack().rename("ret").reset_index()
    w5 = mr.nsmallest(5, "ret")
    months = [f"{int(y)}-{int(m):02d}" for y, m in w5[["year", "month"]].to_numpy()]
    worst = tl[tl["entry_time"].dt.strftime("%Y-%m").isin(months)].copy()
    cols = ["entry_time", "exit_time", "exit_reason", "rr_fill", "d_sl", "d_tp", "width_atr", "sl_atr",
            "departure_height_atr", "profile_changed_since_first_test_start", "profile_changed_since_chain_start",
            "n_rolls", "pattern_name", "sfp", "regime_above_sma200",
            "regime_mom20d", "regime_ret24h", "regime_hours_since_crash", "net_pnl", "return_on_equity", "net_R",
            "ambiguous_exit_sl_and_tp", "ambiguous_exit_sl_and_poc"]
    worst[cols].to_csv(OUT / f"worst_month_trades{sfx}.csv", index=False, float_format="%.10g")
    post = S.post_stop_paths(tl, out["df1h"])
    post.to_csv(OUT / f"post_stop_paths{sfx}.csv", index=False, float_format="%.10g")
    psum = {}
    for H_ in (6, 12, 24):
        m = post[f"window_complete_{H_}h"]
        pp = post[m & post["poc_above_entry"]]
        psum[f"{H_}h"] = {"n_sl": int(m.sum()), "reached_poc": int(pp[f"reached_poc_{H_}h"].sum()), "n_poc_above_entry": int(len(pp)),
                          "reached_vah": int(post.loc[m, f"reached_vah_{H_}h"].sum()),
                          "median_further_drop_R": float(post.loc[m, f"further_drop_below_sl_R_{H_}h"].median()),
                          "share_further_drop_gt_0_5R": float((post.loc[m, f"further_drop_below_sl_R_{H_}h"] < -0.5).mean())}
    summary = {
        "version": version,
        "opportunities": int(len(op)), "status_counts": op["status"].value_counts().to_dict(),
        "second_test_qualified_by_all": op["second_test_qualified_by"].value_counts().to_dict(),
        "second_test_qualified_by_filled": op.loc[op["status"] == "filled", "second_test_qualified_by"].value_counts().to_dict(),
        "filled_profile_changed_since_first_test_start": int(op.loc[op["status"] == "filled", "profile_changed_since_first_test_start"].sum()),
        "filled_profile_changed_since_chain_start": int(op.loc[op["status"] == "filled", "profile_changed_since_chain_start"].sum()),
        "filled_n_rolls": op.loc[op["status"] == "filled", "n_rolls"].value_counts().sort_index().to_dict(),
        "filled_signal_at_profile_supersede_instant": int(op.loc[op["status"] == "filled", "signal_at_profile_supersede_instant"].sum()),
        "filled_regime_sma200_uses_spot_warmup": int(op.loc[op["status"] == "filled", "regime_sma200_uses_spot_warmup"].sum()),
        "filled_profile_window_with_partial_hour": int((op.loc[op["status"] == "filled", "profile_window_n_partial"] > 0).sum()),
        "filled_departure_height_atr_quantiles": op.loc[op["status"] == "filled", "departure_height_atr"].quantile(
            [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).to_dict(),
        "next_open_vs_close_bp_filled": op.loc[op["status"] == "filled", "next_open_vs_close_bp"].describe().to_dict(),
        "boxes": {"n": int(len(bl)), "ever_invalid_while_current": int(bl["invalidated_at"].notna().sum()),
                  "invalid_from_activation": int(bl["invalid_from_activation"].sum()),
                  "own_breakdown_while_current": int((bl["invalidated_at"].notna() & ~bl["invalid_from_activation"]).sum()),
                  "same_type_pivot_pairs": int((bl["pivot_a_type"] == bl["pivot_b_type"]).sum())},
        "trades": int(len(tl)),
        "intrabar_path_ambiguity": {
            "exit_bar_touches_sl_and_tp": int(tl["ambiguous_exit_sl_and_tp"].sum()),
            "exit_bar_touches_sl_and_poc": int(tl["ambiguous_exit_sl_and_poc"].sum()),
            "any": int((tl["ambiguous_exit_sl_and_tp"] | tl["ambiguous_exit_sl_and_poc"]).sum()),
            "share_any": float((tl["ambiguous_exit_sl_and_tp"] | tl["ambiguous_exit_sl_and_poc"]).mean()),
            "entry_and_exit_same_bar": int((tl["exit_bar"] == tl["entry_bar"]).sum()),
            "rule": "worst case: stop-loss first; POC/TP touched inside the exit bar is not counted as reached"},
        "fig1_exclusions": rc["excluded_from_fig1_because"].replace("", "included").value_counts().to_dict(),
        "cost_decomposition_usd": tl[["gross_pnl_ref", "slippage_impact", "commission", "funding_cashflow", "net_pnl"]].sum().to_dict(),
        "worst5_months": months, "worst5_trades": int(len(worst)),
        "worst5_exit_reasons": worst["exit_reason"].value_counts().to_dict(),
        "post_stop": psum,
        "low_rr": S.low_rr_decomposition(tl),
        "slices": sel[["slice_no", "slice_reason", "entry_time", "exit_reason", "rr_fill", "r_multiple"]],
    }
    write_json(OUT / f"ledgers_summary_{version}.json", summary)
    if plots:
        (FIG / "slices").mkdir(parents=True, exist_ok=True)
        sig = out["signals"]
        for _, r in sel.iterrows():
            S.plot_slice(r, out["df1h"], out["df4h"], sig, out["boxes"],
                         FIG / "slices" / f"slice_{int(r['slice_no']):02d}{sfx}.png")
    return {"op": op, "tl": tl, "rc": rc, "sel": sel, "summary": summary}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="B_original")
    a = ap.parse_args()
    r = build(a.version)
    import json
    print(json.dumps({k: v for k, v in r["summary"].items() if k not in ("slices",)}, default=str, ensure_ascii=False, indent=1)[:5000])

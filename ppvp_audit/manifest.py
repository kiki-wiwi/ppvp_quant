"""
ppvp_audit/manifest.py — 冻结 B_original：代码版本、未提交差异、配置、数据清单与哈希、随机种子、交易账本、图表、
复现命令，以及历史上 VAL / OOS / ETH 信息的暴露登记。输出 reports/audit_round/baseline_manifest.json。

B_original = 本轮开始时仓库 HEAD 的策略与引擎代码在 IS 上的结果（BASELINE_STRATEGY / BASELINE_BACKTEST）。
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from config import BASELINE_BACKTEST, BASELINE_STRATEGY, IS_END, IS_START

from .common import OUT, ROOT, SEED, git_state, is_run, sha256_file, write_json

ENGINE_FILES = ("config.py", "pipeline.py", "data_loader.py", "ppvp_engine.py", "strategy_state_machine.py",
                "backtester.py", "metrics.py", "attribution.py", "numba_compat.py")
PHASE2_FIGURES = ("phase2_equity_underwater.png", "phase2_is_exit_paths.png", "phase2_is_null_test.png",
                  "phase2_is_regime.png", "phase2_is_winrate_by_rr.png")

# 历史暴露登记：依据 docs/phase_logs/PHASE0/1/2 的原文整理（提交号为记录中写明的版本）。
EXPOSURE_REGISTER = [
    {"phase": "Phase 0", "commits": "cb95e21", "segment": "IS+VAL+OOS", "symbols": "BTC, ETH",
     "kind": "data quality (no strategy output)",
     "what": "全区间行数、缺口与占位 K 线检测（OOS 内 2026-10-28 归档缺口、VAL 内 2024-10-28 占位 K 线）、"
             "OKX 补数及其相对现货的价差、全样本永续-现货基差与收益相关、2020-2026 资金费率均值与年化成本、"
             "2024-03-01~14 与 OKX 的价格比对、官方 4H 归档错误裁决",
     "influenced_decisions": "数据清洗政策：≥6 小时缺口用 OKX 补齐（只发生在 OOS 的 2026-10-28）、占位 K 线剔除"},
    {"phase": "Phase 1", "commits": "ee26506..d1bc53c", "segment": "IS+VAL (research period)", "symbols": "BTC, ETH",
     "kind": "outcome counts and aggregate equity (debug output)",
     "what": "中间版本的研究期离场原因计数（BTC 止损 85 / 止盈 60，ETH 止损 104 / 止盈 56）；研究期若干配置的期末净值"
             "（显示基线明显亏损）；破位离场计数（22 次中 12 次）；第一轮审计中的亏损止盈交易计数",
     "influenced_decisions": "记录称未据此改动代码或参数（用于核对重构前后一致性）"},
    {"phase": "Phase 1", "commits": "ee26506 (removed in 3e494cf, still in git history)", "segment": "OOS",
     "symbols": "BTC, ETH", "kind": "activity counts",
     "what": "两个 OOS 截断点上的累计信号数与平仓交易数（无收益信息）", "influenced_decisions": "记录称从未用于决策"},
    {"phase": "Phase 1-2", "commits": "ee26506..04724f0", "segment": "IS+VAL+OOS", "symbols": "BTC, ETH",
     "kind": "pass/fail look-ahead gate; structural counts",
     "what": "防未来函数闸门在 2020-2026 全量数据上截断比对（只输出通过 / 失败）；耗时基准记录全量 K 线数与箱体数；"
             "研究期结构计数（箱体 1,579 / 1,558，原始信号 193 / 202，消融表中的可交易 / 成交笔数，INVALID 时间占比）",
     "influenced_decisions": "闸门失败会触发引擎修复（未来函数类缺陷），不涉及绩效"},
    {"phase": "Phase 2", "commits": "a59f783..04724f0", "segment": "VAL", "symbols": "BTC",
     "kind": "performance (headline)",
     "what": "基线 VAL 夏普 −0.525（区间 [−1.93, +1.11]）、累计 −6.9%、最大回撤 −13.59%、30 笔（32 个原始信号）、"
             "胜率 / 盈亏比 / 平均 R、买入持有 VAL 夏普 1.47；6 个成本对照的 VAL 指标；carry / roll 三个变体的 VAL 夏普"
             "（−0.68 至 −0.33）与平均 R；草稿图（a59f783、f85c142）画出过 VAL 日度净值路径；"
             "test_attribution.py 把 VAL 夏普钉为常量",
     "influenced_decisions": "用于 Phase 2 第 3b 节按事先规则否决 carry / roll 变体，以及 Phase 3 判定规则的设计讨论；"
                             "VAL 因此不再是未见数据"},
    {"phase": "Phase 2", "commits": "a59f783..04724f0", "segment": "ETH", "symbols": "ETH",
     "kind": "none beyond structural runs", "what": "参数锁定前只做结构性运行（不回测）；闸门见上一行",
     "influenced_decisions": "无"},
    {"phase": "conversation", "commits": "-", "segment": "IS (figures); VAL headline mentioned", "symbols": "BTC",
     "kind": "screenshots of Phase 2 figures", "what": "用户截图的五张图（IS）；图 5 标题注明 VAL 只报告总体指标",
     "influenced_decisions": "本轮审计的出发点"},
    {"phase": "this round", "commits": "from 04724f0", "segment": "IS only", "symbols": "BTC",
     "kind": "all analyses", "what": "本轮全部分析只用 [2020-01-01, 2024-01-01) 的 BTC 数据",
     "influenced_decisions": "本轮的审计与实验结论"},
]


def freeze() -> dict:
    out = is_run(BASELINE_STRATEGY, BASELINE_BACKTEST)
    res = out["result"]
    led = OUT / "ledgers"
    led.mkdir(parents=True, exist_ok=True)
    tr = res.trades.copy()
    tr.to_csv(led / "trades_B_original_raw.csv", index=False, float_format="%.10g")
    man_data = json.loads((ROOT / "data" / "manifest.json").read_text())
    used = {}
    for name in ("BTCUSDT_perp_1h", "BTCUSDT_funding", "BTCUSDT_spot_1h"):
        meta = man_data["datasets"][name]
        p = ROOT / meta["path"]
        df = pd.read_parquet(p)
        idx = df.index
        used[name] = {"path": meta["path"], "sha256_recorded": meta["sha256"], "sha256_actual": sha256_file(p),
                      "rows_total": int(len(df)), "first": str(idx.min()), "last": str(idx.max()),
                      "rows_used_in_this_round": {"BTCUSDT_perp_1h": int(((idx >= IS_START) & (idx < IS_END)).sum()),
                                                  "BTCUSDT_funding": int(((idx >= IS_START) & (idx <= IS_END)).sum()),
                                                  "BTCUSDT_spot_1h": int((idx < IS_START).sum())}[name],
                      "use": {"BTCUSDT_perp_1h": "全部策略计算（截断在 IS_END 之前）",
                              "BTCUSDT_funding": "资金费现金流（结算时刻 ≤ IS_END）",
                              "BTCUSDT_spot_1h": "只用 2020 年之前的日线收盘做慢速价格指标预热（不含成交量）"}[name],
                      "provider": meta.get("provider"), "gap_fills_in_file": meta.get("gap_fills"),
                      "dropped_placeholder_bars_in_file": meta.get("dropped_placeholder_bars")}
    m1 = json.loads((ROOT / "data" / "manifest_1m_is.json").read_text())
    rep = out["report"]["IS"]
    manifest = {
        "name": "B_original",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "definition": "BASELINE_STRATEGY / BASELINE_BACKTEST on the repository code at the commit below, BTCUSDT perp, "
                      "IS [2020-01-01, 2024-01-01) UTC, data truncated at IS_END before entering the engine",
        "git": git_state(),
        "engine_source_sha256": {f: sha256_file(ROOT / f) for f in ENGINE_FILES},
        "configs": {"strategy": asdict(BASELINE_STRATEGY), "strategy_key": BASELINE_STRATEGY.key(),
                    "backtest": asdict(BASELINE_BACKTEST), "backtest_key": BASELINE_BACKTEST.key()},
        "segment": {"name": "IS", "start": str(IS_START), "end_exclusive": str(IS_END)},
        "data": {"manifest_path": "data/manifest.json", "manifest_sha256": sha256_file(ROOT / "data" / "manifest.json"),
                 "datasets_used": used,
                 "one_minute": {"manifest_path": "data/manifest_1m_is.json", "rows": m1["rows"],
                                "parquet_sha256": m1["parquet_sha256"], "files": len(m1["files"]),
                                "daily_replacements": m1["daily_replacements"],
                                "use": "只用于同根 K 线路径判定、成交量分布近似核对与 C1 限价单成交判定"}},
        "seeds": {"round_master_seed": SEED, "phase2_seed_used_for_figure_reproduction": 20261104},
        "is_headline": {k: rep[k] for k in ("sharpe", "sortino", "cagr", "total_return", "max_dd", "calmar", "trades",
                                            "win_rate", "profit_factor", "avg_r", "avg_hold_bars", "fees", "funding",
                                            "exposure", "exit_reasons", "trades_open_at_end")},
        "ledger": {"path": "reports/audit_round/ledgers/trades_B_original_raw.csv", "rows": int(len(tr)),
                   "sha256": sha256_file(led / "trades_B_original_raw.csv")},
        "phase2_figures": {f: sha256_file(ROOT / "reports" / "figures" / f) for f in PHASE2_FIGURES},
        "reproduce": ["python -m ppvp_audit.data_1m --build   # 只在需要 1m 数据时；下载 IS 的 48 个月度归档并校验",
                      "python -m ppvp_audit.manifest           # 本文件",
                      "python -m ppvp_audit.repro_figures      # 五张图的核验版与 figure_checks.json"],
        "exposure_register": EXPOSURE_REGISTER,
    }
    write_json(OUT / "baseline_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    m = freeze()
    print(m["git"]["describe"], m["is_headline"]["sharpe"], m["ledger"])

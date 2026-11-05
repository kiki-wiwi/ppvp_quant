"""
ppvp_audit/b_audited.py — B_audited 的定义与逐位一致性核验（IS only）。

B_audited = 当前工作树的代码（相对 B_original 只修复已证实的实现错误，见 B_AUDITED_FIXES）+ 与 B_original 完全相同的
配置。核验方法：在 B_original 冻结提交（baseline_manifest.json 的 git.commit）的独立工作树中，与在当前工作树中，
各自用子进程运行 IS 基线（ppvp_audit.common.is_run），对 1H / 4H、箱体表、信号表、成交表、逐小时净值与持仓逐列计算
指纹并比较；任何一列不同都会列出。两边都只读 IS 截断后的数据。

用法：python -m ppvp_audit.b_audited
输出：reports/audit_round/b_audited_check.json
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .common import OUT, ROOT, git_state, write_json

# 修复清单（审计报告第 6 节）：只有被证实的实现错误；策略与定义问题一律不在此列
B_AUDITED_FIXES = [
    {"id": "D-1", "file": "data_loader.py", "kind": "implementation error (latent; IS unaffected)",
     "what": "缺口补齐会把清洗阶段剔除的维护占位小时当作归档遗漏，用 OKX 数据补回，违反“维护期保持缺失”的成文规则",
     "fix": "占位小时不进入补齐候选（fill_archive_gaps(..., placeholders=...)）", "test": "test_audit_round.py::test_gap_fill_keeps_placeholder_maintenance_missing"},
    {"id": "D-5", "file": "data_loader.py", "kind": "implementation error (latent; IS unaffected)",
     "what": "缺口位于数据开头时成交量缩放比例样本为空，补出的成交量为 NaN，进而使箱体的成交量分布全为 NaN",
     "fix": "重叠样本少于 MIN_SCALE_HOURS=24 小时或比例非有限时直接报错", "test": "test_audit_round.py::test_gap_fill_refuses_without_pre_gap_overlap"},
    {"id": "D-8", "file": "data_loader.py", "kind": "process safeguard (section 9 constraint)",
     "what": "缓存缺失时静默联网重建全部缓存（含 1m 日度归档的裁决）",
     "fix": "缓存缺失时报 FileNotFoundError，只有显式 python data_loader.py --build 才联网", "test": "test_audit_round.py::test_missing_cache_raises_instead_of_rebuilding"},
    {"id": "E-1", "file": "ppvp_audit/engine_path.py", "kind": "implementation error (latent; IS unaffected)",
     "what": "路径撮合器按子 K 线下标而不是“入场时刻早于结算时刻”判断资金费，数据缺口内的结算会被漏收",
     "fix": "按时间戳判断（entry_t < fundingTime），两次结算映射到同一根 K 线时报错", "test": "test_audit_round.py::test_engine_path_charges_funding_inside_a_gap_like_the_backtester"},
    {"id": "E-2", "file": "ppvp_audit/engine_path.py", "kind": "implementation error (latent; IS unaffected)",
     "what": "路径撮合器静默忽略它不支持的回测参数（现货场所、不复利、以损定仓、时间止损、破位离场、关闭资金费）",
     "fix": "不支持的配置直接报错；include_funding=False 时按配置不计资金费", "test": "test_audit_round.py::test_engine_path_rejects_unsupported_configs_and_honours_funding_switch"},
]

_FINGERPRINT = r'''
import hashlib, json, sys
import numpy as np
import pandas as pd
from config import BASELINE_BACKTEST, BASELINE_STRATEGY
from ppvp_audit.common import is_run

def col_hash(s):
    s = pd.Series(s)
    try:
        b = pd.util.hash_pandas_object(s, index=False).to_numpy().tobytes()
    except TypeError:
        b = s.astype(str).str.cat(sep="\x1f").encode()
    return hashlib.sha256(str(s.dtype).encode() + b).hexdigest()

def frame_fp(df):
    if isinstance(df, pd.Series):
        df = df.to_frame(df.name if df.name is not None else "value")
    cols = {str(c): col_hash(df[c]) for c in df.columns}
    idx = col_hash(pd.Series(df.index))
    tot = hashlib.sha256((idx + "".join(f"{k}:{v}" for k, v in cols.items())).encode()).hexdigest()
    return {"rows": int(len(df)), "index": idx, "columns": cols, "sha256": tot}

out = is_run(BASELINE_STRATEGY, BASELINE_BACKTEST)
res = out["result"]
fp = {k: frame_fp(v) for k, v in (("df1h", out["df1h"]), ("df4h", out["df4h"]), ("boxes", out["boxes"]),
                                  ("signals", out["signals"]), ("trades", res.trades),
                                  ("equity", res.equity), ("position", res.position))}
fp["is_headline"] = {k: out["report"]["IS"][k] for k in ("sharpe", "total_return", "max_dd", "trades", "fees", "funding")}
fp["strategy_key"] = BASELINE_STRATEGY.key()
fp["backtest_key"] = BASELINE_BACKTEST.key()
print("FINGERPRINT_JSON=" + json.dumps(fp, default=float))
'''


def _fingerprint(root: Path) -> dict:
    p = subprocess.run([sys.executable, "-c", _FINGERPRINT], cwd=root, capture_output=True, text=True, timeout=3600)
    if p.returncode != 0:
        raise RuntimeError(f"fingerprint run failed in {root}:\n{p.stderr[-4000:]}")
    line = [x for x in p.stdout.splitlines() if x.startswith("FINGERPRINT_JSON=")]
    if len(line) != 1:
        raise RuntimeError(f"no fingerprint in output of {root}")
    return json.loads(line[0][len("FINGERPRINT_JSON="):])


def _diff(a: dict, b: dict) -> dict:
    out = {}
    for k in ("df1h", "df4h", "boxes", "signals", "trades", "equity", "position"):
        fa, fb = a[k], b[k]
        cols_a, cols_b = set(fa["columns"]), set(fb["columns"])
        out[k] = {"identical": fa["sha256"] == fb["sha256"], "rows": [fa["rows"], fb["rows"]],
                  "index_identical": fa["index"] == fb["index"],
                  "columns_only_in_B_original": sorted(cols_a - cols_b), "columns_only_in_B_audited": sorted(cols_b - cols_a),
                  "columns_differing": sorted(c for c in cols_a & cols_b if fa["columns"][c] != fb["columns"][c])}
    return out


def run() -> dict:
    man = json.loads((OUT / "baseline_manifest.json").read_text())
    commit = man["git"]["commit"]
    tmp = Path(tempfile.mkdtemp(prefix="ppvp_b_original_"))
    wt = tmp / "wt"
    try:
        subprocess.run(["git", "worktree", "add", "--detach", str(wt), commit], cwd=ROOT, check=True,
                       capture_output=True, text=True)
        fa = _fingerprint(wt)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=ROOT, capture_output=True, text=True)
        shutil.rmtree(tmp, ignore_errors=True)
    fb = _fingerprint(ROOT)
    diff = _diff(fa, fb)
    changed = subprocess.run(["git", "diff", "--stat", commit, "--", "."], cwd=ROOT, capture_output=True, text=True).stdout
    rep = {
        "name": "B_audited",
        "definition": "current working tree code (only proven implementation-error fixes relative to B_original, listed "
                      "below) with the B_original configs; BTCUSDT perp IS [2020-01-01, 2024-01-01)",
        "b_original_commit": commit,
        "b_audited_git": git_state(),
        "diff_stat_vs_b_original_commit": changed,
        "fixes": B_AUDITED_FIXES,
        "configs_identical": fa["strategy_key"] == fb["strategy_key"] and fa["backtest_key"] == fb["backtest_key"],
        "strategy_key": fb["strategy_key"], "backtest_key": fb["backtest_key"],
        "comparison": diff,
        "all_identical": all(v["identical"] for v in diff.values()),
        "is_headline_B_original": fa["is_headline"], "is_headline_B_audited": fb["is_headline"],
        "fingerprints": {"B_original": {k: fa[k]["sha256"] for k in diff}, "B_audited": {k: fb[k]["sha256"] for k in diff}},
        "reproduce": "python -m ppvp_audit.b_audited",
    }
    write_json(OUT / "b_audited_check.json", rep)
    return rep


if __name__ == "__main__":
    r = run()
    print(json.dumps({"all_identical": r["all_identical"], "configs_identical": r["configs_identical"],
                      "comparison": {k: v["identical"] for k, v in r["comparison"].items()},
                      "differing": {k: v["columns_differing"] for k, v in r["comparison"].items() if not v["identical"]}},
                     indent=1))

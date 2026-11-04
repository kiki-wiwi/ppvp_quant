"""
tools/mutation_test.py — 突变测试：证明测试套件对已知缺陷类别有检出力。

每个突变体是对一份代码副本的最小文本修改（old → new，old 必须在目标文件中恰好出现一次）：
* L*：未来函数 / 执行价位泄露 —— 必须被防未来函数闸门（test_lookahead.py）拦截；
* S*：状态机规则回退 —— 必须被回归测试（test_audit_regressions.py）拦截；
* E*：执行 / 指标 / 管道规则回退 —— 必须被回归测试拦截。
对照组（control，不做修改）必须两套测试全部通过。

用法（在仓库根目录）：
    python tools/mutation_test.py                 # 全部突变体（约 30 分钟）
    python tools/mutation_test.py --check         # 只检查每个修改点在当前代码中唯一存在
    python tools/mutation_test.py L1 S3 ...       # 指定突变体（只打印结果）
    python tools/mutation_test.py --merge S15 ... # 重跑指定突变体并合并进已有的结果文件（测试文件只改了一部分时使用）
结果写入 reports/mutation_results.json（只记录是否被拦截与首个失败测试的节点名，不记录任何失败消息）。
副本建在系统临时目录（或 --workdir），data/ 以符号链接接入，每个突变体使用独立的 Numba 缓存目录。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GATE = ["test_lookahead.py"]
REG = ["test_audit_regressions.py"]
SM, BT, ENG, MET, PIPE = "strategy_state_machine.py", "backtester.py", "ppvp_engine.py", "metrics.py", "pipeline.py"

# id: (说明, [(文件, old, new), ...], 应拦截它的测试文件)
MUTANTS: dict[str, tuple[str, list[tuple[str, str, str]], list[str]]] = {
    # ------------------------------------------------------------------ 未来函数 / 执行价位（闸门）
    "L1": ("入场按信号 K 线自身的开盘价成交（提前 1 小时）",
           [(BT, "efill = c[i] * (1.0 + slip)", "efill = o[i] * (1.0 + slip)")], GATE),
    "L2": ("资金费按结算 K 线收盘价计价（晚 1 小时）",
           [(BT, "f = qty * o[i] * fr\n", "f = qty * c[i] * fr\n")], GATE),
    "L3": ("Pinbar 判定偷看下一根收盘",
           [(SM, '"pin": bullish_pinbar(o, h, l, c, cfg), "eng"',
             '"pin": bullish_pinbar(o, h, l, c, cfg) & np.concatenate([c[1:] > c[:-1], [False]]), "eng"')], GATE),
    "L4": ("成交量均线居中滚动（使用未来约 10 根）",
           [(SM, "pd.Series(v).rolling(cfg.vol_sma_len, min_periods=cfg.vol_sma_len).mean()",
             "pd.Series(v).rolling(cfg.vol_sma_len, min_periods=1, center=True).mean()")], GATE),
    "L5": ("箱体在确认前 2 小时即被使用",
           [(ENG, 'np.searchsorted(avail, to_ns(df1h.index), side="right") - 1',
             'np.searchsorted(avail, to_ns(df1h.index) + np.int64(2 * 3600 * 10**9), side="right") - 1')], GATE),
    "L6": ("日线宏观过滤在当日开盘即可用",
           [(SM, "avail = to_ns(d.index) + np.int64(86400 * 10**9)", "avail = to_ns(d.index)")], GATE),
    "L7": ("枢轴确认滞后少 1 根 4H",
           [(ENG, '"available_time": avail4[b + R],', '"available_time": avail4[b + R - 1],')], GATE),
    "L8": ("成交量分布窗口越过箱体确认时刻",
           [(ENG, "t0, t1 = open4[a], avail4[b]  ", "t0, t1 = open4[a], avail4[min(b + R + 1, len(avail4) - 1)]  ")], GATE),
    "L9": ("破位离场读取下一根的形态与收盘",
           [(BT, "bd_hit = exit_on_brk and bodyvol[i] and c[i] < lv_brk[i]",
             "bd_hit = exit_on_brk and i + 1 < n and bodyvol[i + 1] and c[i + 1] < lv_brk[i]")], GATE),
    "L10": ("破位阈值读取下一根的 ATR（记录的价位随之改变）",
            [(BT, "brk_lvl = val_e - k1atr[i]", "brk_lvl = val_e - k1atr[min(i + 1, n - 1)]")], GATE),
    "L11": ("结算时刻收盘离场漏计资金费",
            [(BT, "fr2 = fund_close[i]", "fr2 = np.nan")], GATE),
    "L12": ("破位阈值读取两根之后的 ATR（记录的价位随之改变）",
            [(BT, "brk_lvl = val_e - k1atr[i]", "brk_lvl = val_e - k1atr[min(i + 2, n - 1)]")], GATE),
    "L13": ("下一根将触及止损时偷偷下移止损",
            [(BT, "            lv_stop[i] = sl\n",
              "            if i + 1 < n and l[i + 1] <= sl:\n                sl = sl * 0.999\n            lv_stop[i] = sl\n")], GATE),
    "L14": ("破位离场判定读取两根之后的 ATR，记录的价位不变（第三轮 M07b）",
            [(BT, "bd_hit = exit_on_brk and bodyvol[i] and c[i] < lv_brk[i]",
              "bd_hit = exit_on_brk and bodyvol[i] and c[i] < val_e - k1atr[min(i + 2, n - 1)]")], GATE),
    "L15": ("破位离场判定读取下一根收盘（第三轮 M04）",
            [(BT, "bd_hit = exit_on_brk and bodyvol[i] and c[i] < lv_brk[i]",
              "bd_hit = exit_on_brk and bodyvol[i] and c[min(i + 1, n - 1)] < lv_brk[i]")], GATE),
    "L16": ("破位离场判定读取下一根的放量实体条件（第三轮 M05）",
            [(BT, "bd_hit = exit_on_brk and bodyvol[i] and c[i] < lv_brk[i]",
              "bd_hit = exit_on_brk and bodyvol[min(i + 1, n - 1)] and c[i] < lv_brk[i]")], GATE),
    "L17": ("下一根最高价将触及 POC 时提前在本根末把止损上移到保本价（第三轮 M12）",
            [(BT, "            if qty == 0.0:\n                t_exit_bar[k] = i\n",
              "            if qty > 0.0 and tp_mode == 1 and move_be and (not partial_done) and h[min(i + 1, n - 1)] >= poc:\n"
              "                sl = entry_px\n            if qty == 0.0:\n                t_exit_bar[k] = i\n")], GATE),
    "L18": ("本根开始时按本根最高价是否触及 POC 决定保本止损（第三轮 M28）",
            [(BT, "            lv_stop[i] = sl\n",
              "            if tp_mode == 1 and move_be and (not partial_done) and h[i] >= poc:\n                sl = entry_px\n"
              "            lv_stop[i] = sl\n")], GATE),
    "L19": ("本根末按 min(本根低点, 下一根低点) 收紧止损（第三轮 M34）",
            [(BT, "            if qty == 0.0:\n                t_exit_bar[k] = i\n",
              "            if qty > 0.0:\n                sl = max(sl, min(l[i], l[min(i + 1, n - 1)]) - 3.0 * k1atr[i])\n"
              "            if qty == 0.0:\n                t_exit_bar[k] = i\n")], GATE),
    "L20": ("POC 分批由下一根最高价触发（第三轮 M03）",
            [(BT, "poc < tp and h[i] >= poc:", "poc < tp and h[min(i + 1, n - 1)] >= poc:")], GATE),
    "L21": ("收盘离场的结算费率经下一根的开盘映射读取（第三轮 M11）",
            [(BT, "fr2 = fund_close[i]", "fr2 = fund_open[min(i + 1, n - 1)]")], GATE),
    "L22": ("数据首日尚无完整永续日线时不做现货预热（截断与全量的宏观状态不一致，第三轮 EX3-8）",
            [(SM, "    if spot_daily is not None:\n        # 预热", "    if spot_daily is not None and len(d):\n        # 预热")], GATE),
    # ------------------------------------------------------------------ 状态机规则回退（回归测试）
    "S1": ("放量跌穿记忆只在 INVALID 状态下检查（SM-R1 回退）",
           [(SM, "                hv_block = False\n                if not np.isnan(a):",
             "                hv_block = False\n                if st == 5 and not np.isnan(a):")], REG),
    "S2": ("EXHAUSTED 期间的放量跌穿被忽略（SM-R2 回退）",
           [(SM, "        if st == 5 or st == 6:\n            if is_brk:\n", "        if st == 5 or st == 6:\n            if is_brk and st == 5:\n")], REG),
    "S3": ("箱体切换时不认领任何未完成的“先到带上方”要求（SM-R3 回退）",
           [(SM, "nd_cl[e] = (not np.isnan(a)) and abs(box_val[b] - nd_val[e]) <= ov", "nd_cl[e] = False")], REG),
    "S4": ("“全新结构”边界取跌穿 K 线开盘而非收盘",
           [(SM, "            hv_ts[hcnt] = open_ns[i] + hour\n            hcnt += 1\n            state[i] = st",
             "            hv_ts[hcnt] = open_ns[i]\n            hcnt += 1\n            state[i] = st")], REG),
    "S5": ("失效期间再次跌穿不后移记忆时刻",
           [(SM, "                                hv_ts[e] = open_ns[i] + hour   # 同一支撑区",
             "                                pass   # 同一支撑区")], REG),
    "S6": ("激活时一律要求先到带上方（取消带内激活例外）",
           [(SM, "                    if below and not need_above:\n", "                    if not need_above:\n")], REG),
    "S7": ("激活判定阈值误用带上沿",
           [(SM, "zl_new = box_val[b] - k1 * a", "zl_new = box_val[b] + k2 * a")], REG),
    "S8": ("滚动后 carry 基准不更新为第 2 次测试的 VAL",
           [(SM, "                    seq_val = t2_val\n", "                    seq_val = seq_val\n")], REG),
    "S9": ("激活即否决时写入放量跌穿记忆",
           [(SM, "                        st = 5                             # 否决该箱体；否决本身不写入放量跌穿记忆\n",
             "                        st = 5\n                        hv_vals[hcnt] = box_val[b]\n"
             "                        hv_ts[hcnt] = open_ns[i]\n                        hcnt += 1\n")], REG),
    "S10": ("价值区扩展在两侧无桶时不停止（SM-R4 回退）",
            [(ENG, "        if not (can_up or can_dn):\n            break", "        if False:\n            break")], REG),
    "S11": ("放量跌穿记忆从不阻断（SPEC-1 完全回退）",
            [(SM, "                        if abs(box_val[b] - hv_vals[e]) <= ov:\n                            hv_block = True",
              "                        if False and abs(box_val[b] - hv_vals[e]) <= ov:\n                            hv_block = True")], REG),
    "S12": ("信号后不要求先脱离到带上方（SPEC-3 回退）",
            [(SM, "                nd_t[ncnt] = open_ns[i] + hour\n                nd_cl[ncnt] = True\n                ncnt += 1\n                need_above = True\n",
              "                need_above = False\n")], REG),
    "S13": ("持仓期破位参照当前箱体（EXECUTION-1 回退）",
            [(BT, "brk_lvl = val_e - k1atr[i]", "brk_lvl = val_arr[i] - k1atr[i]")], REG),
    "S14": ("放量跌穿记忆退化为单条（新写入覆盖旧记忆）",
            [(SM, "            st = 5\n            hv_vals[hcnt] = box_val[cur]\n", "            st = 5\n            hcnt = 0\n            hv_vals[hcnt] = box_val[cur]\n")], REG),
    "S15": ("放量跌穿记忆容量 8 条、满时覆盖最后一条（第三轮 SM3-2 回退）",
            [(SM, "            hv_vals[hcnt] = box_val[cur]\n            hv_ts[hcnt] = open_ns[i] + hour\n            hcnt += 1\n            state[i] = st",
              "            hcnt = min(hcnt, 7)\n            hv_vals[hcnt] = box_val[cur]\n            hv_ts[hcnt] = open_ns[i] + hour\n"
              "            hcnt += 1\n            state[i] = st")], REG),
    "S16": ("阻断只检查第一条记忆（第三轮 M09 / DOC3-2 M5）",
            [(SM, "                    for e in range(hcnt):\n                        if abs(box_val[b] - hv_vals[e]) <= ov:",
              "                    for e in range(min(hcnt, 1)):\n                        if abs(box_val[b] - hv_vals[e]) <= ov:")], REG),
    "S17": ("INVALID 期间其他支撑区（如被否决箱体）的放量跌穿不写入记忆（第三轮 M14 / DOC3-2 M4）",
            [(SM, "                if not same:\n", "                if (not same) and st == 6:\n")], REG),
    "S18": ("再次跌穿只后移第一条同区记忆（第三轮 M31 / DOC3-2 M3）",
            [(SM, "                                hv_ts[e] = open_ns[i] + hour   # 同一支撑区再次放量跌穿被跌穿的带下沿：记忆时刻后移\n",
              "                                hv_ts[e] = open_ns[i] + hour\n                                break\n")], REG),
    "S19": ("放量跌穿记忆永不失效（DOC3-2 M1a）",
            [(SM, "                    if box_wstart[b] >= hv_ts[e]:\n", "                    if False:\n")], REG),
    "S20": ("离开 INVALID 时丢弃全部“先到带上方”要求（第三轮 SM3-1 回退：失效绕行）",
            [(SM, "                    # 新箱体认领同区的全部未完成“须先到带上方”要求",
              "                    if st == 5:\n                        ncnt = 0\n                    # 新箱体认领同区的全部未完成“须先到带上方”要求")], REG),
    "S21": ("换到其他支撑区时丢弃全部未认领的要求（第三轮 SM3-1 回退：A→B→A 绕行）",
            [(SM, "                    if (not nd_cl[e]) and box_wstart[b] >= nd_t[e]:\n", "                    if not nd_cl[e]:\n")], REG),
    "S22": ("只有 ARMED 状态下到带上方才清除要求（第三轮 SM3-1：失效期间的脱离被忽略）",
            [(SM, "        if ncnt > 0:                                       # 整根位于", "        if ncnt > 0 and st == 1:                           # 整根位于")], REG),
    "S23": ("被认领的要求也因全新结构失效",
            [(SM, "                    if (not nd_cl[e]) and box_wstart[b] >= nd_t[e]:\n", "                    if box_wstart[b] >= nd_t[e]:\n")], REG),
    "S24": ("未认领要求的失效边界用 > 而非 ≥",
            [(SM, "                    if (not nd_cl[e]) and box_wstart[b] >= nd_t[e]:\n", "                    if (not nd_cl[e]) and box_wstart[b] > nd_t[e]:\n")], REG),
    "S25": ("价值区单侧扩展时越界回绕到另一端（第三轮 SM3-4 M34）",
            [(ENG, "if can_up and ((not can_dn) or hist[hi + 1] >= hist[lo - 1] - tol):", "if can_up and (hist[hi + 1] >= hist[lo - 1] - tol):")], REG),
    "S26": ("不检查箱体窗口起点单调（第三轮 SM3-6）",
            [(SM, "    if nb > 1 and (np.diff(bws) < 0).any():", "    if False:")], REG),
    # ------------------------------------------------------------------ 执行 / 指标 / 管道规则回退（回归测试）
    "E1": ("资金费网格用 np.arange 浮点估长（长窗口丢掉终点，第三轮 DOC3-1 回退）",
           [(BT, "    grid = np.int64(first) + np.int64(step) * np.arange(n_pts, dtype=np.int64)",
             "    grid = np.arange(np.int64(first), np.int64(cln) + 1, np.int64(step), dtype=np.int64)")], REG),
    "E2": ("接受不在结算网格上的时间戳（第三轮 EX3-4 回退）",
           [(BT, "    if off.any():\n", "    if False:\n")], REG),
    "E3": ("交易统计不检查 OOS 离场（第三轮 EX3-5 回退）",
           [(MET, "    _check_trades_extent(tr, allow_oos)\n    entries =", "    entries =")], REG),
    "E4": ("不报告跨越段起点的截断持仓（第三轮 EX3-6 回退）",
           [(MET, "            capped_open_at_start = int((straddle & tr_all[\"capped\"]).sum())",
             "            capped_open_at_start = 0")], REG),
    "E5": ("缓存返回值与缓存共享行索引（第三轮 DOC3-6 回退）",
           [(PIPE, "    y.index = x.index.copy(deep=True)\n", "    y.index = x.index\n")], REG),
    "E6": ("BacktestResult 的 repr 显示数值（第三轮 EX3-7 回退）",
           [(BT, "@dataclass(repr=False)\nclass BacktestResult:", "@dataclass\nclass BacktestResult:"),
            (BT, "    def __repr__(self) -> str:\n        # 只显示", "    def _repr_unused(self) -> str:\n        # 只显示")], REG),
}


def _apply(work: Path, edits) -> str | None:
    for fn, old, new in edits:
        src = (work / fn).read_text()
        if src.count(old) != 1:
            return f"pattern occurs {src.count(old)} times in {fn}"
        (work / fn).write_text(src.replace(old, new))
    return None


def _copy_repo(dst: Path) -> None:
    dst.mkdir(parents=True)
    for f in REPO.glob("*.py"):
        shutil.copy(f, dst / f.name)
    (dst / "data").symlink_to(REPO / "data")
    (dst / "reports").mkdir()


def run(names, workdir: Path, check_only: bool) -> dict:
    results = {}
    for name in names:
        desc, edits, tests = MUTANTS.get(name, ("对照组：未修改的代码", [], GATE + REG))
        work = workdir / name
        shutil.rmtree(work, ignore_errors=True)
        _copy_repo(work)
        err = _apply(work, edits)
        if err or check_only:
            results[name] = {"description": desc, "applied": err is None, "error": err}
            print(f"{name:8s} {'OK' if err is None else 'PATTERN ERROR: ' + err}", flush=True)
            shutil.rmtree(work, ignore_errors=True)
            continue
        env = dict(os.environ, NUMBA_CACHE_DIR=str(work / ".numba_cache"), PYTHONDONTWRITEBYTECODE="1")
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rf"] + ([] if name == "control" else ["-x"])
        t0 = time.time()
        try:
            r = subprocess.run(cmd + tests, cwd=work, env=env, capture_output=True, text=True, timeout=3600)
            out, code = r.stdout, r.returncode
        except subprocess.TimeoutExpired:
            out, code = "", -1
        failed = [ln.split(" ")[1] for ln in out.splitlines() if ln.startswith("FAILED ")]
        summary = next((ln for ln in reversed(out.splitlines()) if (" passed" in ln or " failed" in ln)), "")
        results[name] = {"description": desc, "applied": True, "tests": tests,
                         "detected": code != 0, "first_failing_test": failed[0] if failed else None,
                         "summary": summary.strip("= ").split(" in ")[0], "seconds": round(time.time() - t0, 1)}
        print(f"{name:8s} {'KILLED ' if code != 0 else 'SURVIVED'} {results[name]['first_failing_test'] or ''} "
              f"[{results[name]['summary']}]", flush=True)
        shutil.rmtree(work, ignore_errors=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    names = a.names or (["control"] + list(MUTANTS))
    workdir = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="ppvp_mut_"))
    res = run(names, workdir, a.check)
    out = REPO / "reports" / "mutation_results.json"
    if a.merge and not a.check:
        merged = json.loads(out.read_text())
        merged.update(res)
        order = ["control"] + list(MUTANTS)
        out.write_text(json.dumps({k: merged[k] for k in order if k in merged}, indent=2, ensure_ascii=False))
        print(f"merged into {out}")
    elif not a.check and not a.names:
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"written {out}")
    elif not a.check:
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

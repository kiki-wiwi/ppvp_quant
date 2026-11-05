"""
ppvp_audit/sm_trace.py — 状态机追踪副本：由 strategy_state_machine._run_state_machine 的源码派生，逻辑一行不改，
只在信号处额外记录本次两次测试序列的起点（审计 P-6）。关闭追踪不需要：原 15 个输出必须与生产版本逐位一致（tests 钉住）。

额外输出（只在信号 K 线上有值）：
* seq_start：发出信号的序列中“第 1 次测试”开始的 K 线（ARMED → TOUCH1；若经过 roll，则为被滚动成第 1 次测试的那次失败第 2 次测试的起点）；
* chain_start：整条链最初 ARMED → TOUCH1 的 K 线（roll 不改变它）；
* n_rolls：链内 roll 的次数。
"""
from __future__ import annotations

import inspect

import numpy as np

import strategy_state_machine as ssm

_FN = None
N_EXTRA = 3


def _patched_source() -> str:
    src = inspect.getsource(ssm._run_state_machine.py_func)
    assert src.startswith("@njit(cache=True)\n")
    src = src[len("@njit(cache=True)\n"):].replace("def _run_state_machine(", "def _run_state_machine_trace(", 1)

    def rep(old, new):
        nonlocal src
        if src.count(old) != 1:
            raise RuntimeError(f"trace patch site not unique: {old[:60]!r}")
        src = src.replace(old, new)

    rep("    s_need = np.zeros(n, np.bool_)\n",
        "    s_need = np.zeros(n, np.bool_)\n    x_seq = np.full(n, -1, np.int64)\n    x_chain = np.full(n, -1, np.int64)\n"
        "    x_roll = np.zeros(n, np.int64)\n    seq_start = -1\n    chain_start = -1\n    nroll = 0\n")
    rep("                if hv_block or not keep:\n                    t1_low = np.nan\n",
        "                if hv_block or not keep:\n                    seq_start = -1\n                    chain_start = -1\n"
        "                    nroll = 0\n                    t1_low = np.nan\n")
    rep("            elif touched:\n                st = 2\n                t1_low = l[i]\n",
        "            elif touched:\n                st = 2\n                seq_start = i\n                chain_start = i\n"
        "                nroll = 0\n                t1_low = l[i]\n")
    rep("                if roll:\n                    st = 3\n",
        "                if roll:\n                    seq_start = t2_start\n                    nroll += 1\n                    st = 3\n")
    rep("                s_t2_start[i] = t2_start\n",
        "                s_t2_start[i] = t2_start\n                x_seq[i] = seq_start\n                x_chain[i] = chain_start\n"
        "                x_roll[i] = nroll\n                seq_start = -1\n                chain_start = -1\n                nroll = 0\n")
    rep("            s_box, s_need)\n", "            s_box, s_need, x_seq, x_chain, x_roll)\n")
    return src


def trace_fn():
    global _FN
    if _FN is None:
        from numba_compat import njit
        ns = {"np": np}
        exec(compile(_patched_source(), "<sm_trace>", "exec"), ns)
        _FN = njit(cache=False)(ns["_run_state_machine_trace"])
    return _FN


def trace(df1h, scfg, boxes) -> dict:
    """返回原 15 个输出与 3 个追踪输出。"""
    args = ssm.state_machine_args(df1h, scfg, boxes)
    out = trace_fn()(*args)
    return {"core": out[:-N_EXTRA], "seq_start": out[-3], "chain_start": out[-2], "n_rolls": out[-1], "args": args}

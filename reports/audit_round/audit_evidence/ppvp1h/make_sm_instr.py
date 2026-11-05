"""Builds sm_instr.py: a copy of strategy_state_machine._run_state_machine (pure Python) with extra outputs
recording, for each signal, the bar at which the signalling sequence's first test started (ARMED->TOUCH1, or the
start of a failed second test that was rolled into the first test) and the box active there. The logic lines are
unchanged; s05 verifies that all 15 original outputs are bit-identical to the engine."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "sm_copy_raw.py").read_text()
assert src.startswith("@njit(cache=True)\n")
src = src.replace("@njit(cache=True)\n", "", 1)
src = src.replace("def _run_state_machine(", "def run_sm_instr(", 1)


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (old, src.count(old))
    src = src.replace(old, new)


rep("    s_need = np.zeros(n, np.bool_)\n",
    "    s_need = np.zeros(n, np.bool_)\n"
    "    s_seq_start = np.full(n, -1, np.int64)\n"
    "    s_seq_box = np.full(n, -1, np.int64)\n"
    "    s_seq_val = np.full(n, np.nan)\n"
    "    s_t2_val = np.full(n, np.nan)\n"
    "    s_carry = np.zeros(n, np.bool_)\n"
    "    seq_start = -1\n"
    "    seq_box = -1\n"
    "    s_chain_start = np.full(n, -1, np.int64)\n"
    "    s_nroll = np.zeros(n, np.int64)\n"
    "    chain_start = -1\n"
    "    nroll = 0\n"
    "    s_cdist = np.full(n, np.nan)\n")
# carry marker + reset of the instrumentation on non-carry switches
rep("                if hv_block or not keep:\n                    t1_low = np.nan\n",
    "                s_carry[i] = keep and not hv_block\n"
    "                if cur >= 0 and (st == 2 or st == 3 or st == 4) and not np.isnan(a):\n"
    "                    s_cdist[i] = abs(box_val[b] - seq_val) / a\n"
    "                if hv_block or not keep:\n                    seq_start = -1\n                    seq_box = -1\n                    chain_start = -1\n                    nroll = 0\n                    t1_low = np.nan\n")
# first test starts from ARMED
rep("            elif touched:\n                st = 2\n                t1_low = l[i]\n",
    "            elif touched:\n                st = 2\n                seq_start = i\n                seq_box = cur\n                chain_start = i\n                nroll = 0\n                t1_low = l[i]\n")
# roll: the failed second test becomes the first test
rep("                if roll:\n                    st = 3\n",
    "                if roll:\n                    seq_start = t2_start\n                    seq_box = s_box[t2_start]\n                    nroll += 1\n                    st = 3\n")
# record at signal
rep("                s_t2_start[i] = t2_start\n",
    "                s_t2_start[i] = t2_start\n"
    "                s_seq_start[i] = seq_start\n                s_seq_box[i] = seq_box\n"
    "                s_seq_val[i] = seq_val\n                s_t2_val[i] = t2_val\n"
    "                s_chain_start[i] = chain_start\n                s_nroll[i] = nroll\n"
    "                seq_start = -1\n                seq_box = -1\n                chain_start = -1\n                nroll = 0\n")
rep("            s_box, s_need)\n",
    "            s_box, s_need, s_seq_start, s_seq_box, s_seq_val, s_t2_val, s_carry, s_chain_start, s_nroll, s_cdist)\n")
header = "import numpy as np\n\n\n"
(HERE / "sm_instr.py").write_text(header + src)
print("written sm_instr.py")

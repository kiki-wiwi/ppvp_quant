"""s06: B1-compatibility code-path check on the BASELINE run, using the exact running first-test low recorded by the
independent machine (s05). No B1 kernel is run and no B1 signal subset is counted.

Population: IS bars processed while the machine is in DEPARTED or TOUCH2 (state after box-switch handling, before the
bar's own transitions = 3 or 4) whose LOW pierces the running first-test low by >= 1 tick, or the band low, and whose
CLOSE is at or above max(VAL, first-test low). Question: does any such bar end in INVALID, or end the sequence by
anything other than a signal / a box switch?
"""
import pickle
import sys

sys.path.insert(0, "/home/user/ppvp_quant")
import numpy as np
import pandas as pd

from config import IS_END
from ppvp_audit.engine_path import tick_size

Dd = "/tmp/claude-0/-home-user-ppvp-quant/04a66f10-1e5e-5960-9a9f-ffb5b694e43a/scratchpad/audit_round/state1h"
P = pickle.load(open(f"{Dd}/baseline_out.pkl", "rb"))
df1h, sig = P["df1h"], P["signals"]
assert df1h.index.max() < IS_END
R = np.load(f"{Dd}/s05_running.npz")
t1b, stb, st = R["t1_before"], R["st_before"], R["state"]
assert np.array_equal(st, sig.state.to_numpy())
h, l, c, o = (df1h[k].to_numpy(float) for k in ("high", "low", "close", "open"))
zlo = sig.zone_lo.to_numpy(); val = sig.VAL.to_numpy(); brk = sig.breakdown.to_numpy()
sraw = sig.sig_raw.to_numpy(); sbox = sig.box_id.to_numpy()
tick = tick_size(df1h.index).astype(float)
box_switch = np.r_[False, sbox[1:] != sbox[:-1]]

# sanity: at signal bars, the running first-test low (before the bar) equals the kernel's t1_low output
sb = np.flatnonzero(sraw)
print("running first-test low == kernel t1_low at all", len(sb), "signal bars:", np.array_equal(t1b[sb], sig.t1_low.to_numpy()[sb]))

pop_state = np.isin(stb, (3, 4)) & ~np.isnan(t1b)
pierce = (l <= t1b - tick) | (l < zlo)
close_back = c >= np.fmax(val, t1b)
m = pop_state & pierce & close_back
print("\npopulation (DEPARTED/TOUCH2 bars piercing first-test low or band low intrabar, close >= max(VAL, first-test low)):", int(m.sum()))
print("  pierce first-test low by >=1 tick:", int((m & (l <= t1b - tick)).sum()), "; pierce band low:", int((m & (l < zlo)).sum()))
ct = pd.crosstab(pd.Series(stb[m], name="state_before"), pd.Series(st[m], name="state_after"))
print(ct)
print("  -> INVALID (5):", int((st[m] == 5).sum()), " breakdown flag:", int(brk[m].sum()))
arm = m & (st == 1)
print("  -> ARMED (1):", int(arm.sum()), " of which raw signal or box switch:", int((arm & (sraw | box_switch)).sum()),
      " unexplained:", int((arm & ~sraw & ~box_switch).sum()))
dep = m & (st == 3)
print("  -> DEPARTED (3) [needs low > band top, impossible for a piercing bar unless box switch]:", int(dep.sum()),
      " with box switch:", int((dep & box_switch).sum()))
mer = m & (stb == 3) & (st == 2)
print("  -> merged into TOUCH1 (separation not met; the pierce becomes the new first-test low):", int(mer.sum()))
# the (unique) close-based condition that can ever invalidate inside a sequence
print("\nbreakdown bars in IS:", int(brk.sum()), "; any with close >= band low:", int((brk & (c >= zlo)).sum()),
      "; any with close >= VAL:", int((brk & (c >= val)).sum()))

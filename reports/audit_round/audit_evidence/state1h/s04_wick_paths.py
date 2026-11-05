"""s04: B1-compatibility code-path check on the BASELINE run only (no B1 kernel is run, no B1 signal set is counted).

For every IS bar processed in DEPARTED or TOUCH2 (state before the bar = 3 or 4) or TOUCH1 (2) whose LOW pierces below
the running first-test low by >= 1 tick or below the band low, while its CLOSE is back at or above max(VAL, running
first-test low): record the state after the bar and whether a box switch happened at that bar. Also: every transition
into INVALID from an active state must be a close-based breakdown bar (close < band low) or a box switch (hv_block).
Also: merges that lowered the first-test low (a sweep absorbed into test 1 because separation was not met).
"""
import pickle
import sys

sys.path.insert(0, "/home/user/ppvp_quant")
import numpy as np
import pandas as pd

from config import BASELINE_STRATEGY as CFG, IS_END
from ppvp_audit.engine_path import tick_size

Dd = "/tmp/claude-0/-home-user-ppvp-quant/04a66f10-1e5e-5960-9a9f-ffb5b694e43a/scratchpad/audit_round/state1h"
P = pickle.load(open(f"{Dd}/baseline_out.pkl", "rb"))
df1h, sig, boxes = P["df1h"], P["signals"], P["boxes"]
assert df1h.index.max() < IS_END
h, l, c, o = (df1h[k].to_numpy(float) for k in ("high", "low", "close", "open"))
st = sig.state.to_numpy(); zlo = sig.zone_lo.to_numpy(); zhi = sig.zone_hi.to_numpy()
val = sig.VAL.to_numpy(); brk = sig.breakdown.to_numpy(); sbox = sig.box_id.to_numpy(); sraw = sig.sig_raw.to_numpy()
tick = tick_size(df1h.index).astype(float)
n = len(df1h)

# running first-test low reconstructed from the production state array (state 2 bars extend test 1; a roll
# (4 -> 3) replaces it with the test-2 low; a signal/box reset/INVALID clears it)
t1 = np.full(n, np.nan)        # running t1_low BEFORE bar i is processed
t2 = np.full(n, np.nan)
cur1, cur2 = np.nan, np.nan
prev = 0
for i in range(n):
    t1[i], t2[i] = cur1, cur2
    s = st[i]
    box_sw = i > 0 and sbox[i] != sbox[i - 1]
    if box_sw and not (prev in (2, 3, 4) and s in (2, 3, 4)):
        cur1, cur2 = np.nan, np.nan            # non-carry switch resets the sequence
    if s == 2:
        if prev in (1, 0, 5, 6) or (box_sw and np.isnan(cur1)):
            cur1 = l[i]
        else:
            cur1 = min(cur1, l[i]) if not np.isnan(cur1) else l[i]
        cur2 = np.nan
    elif s == 3:
        if prev == 4:                          # roll
            cur1, cur2 = cur2, np.nan
    elif s == 4:
        cur2 = l[i] if prev == 3 else min(cur2, l[i])
    elif s == 1 and sraw[i]:
        cur1, cur2 = np.nan, np.nan
    elif s in (1, 5, 6):
        cur1, cur2 = np.nan, np.nan
    prev = s
# sanity: at raw-signal bars the reconstructed values must equal the kernel's outputs
sb = np.flatnonzero(sraw)
t1_at_sig = sig.t1_low.to_numpy()[sb]
# t1[i] is the value before bar i; t1 is unchanged by a TOUCH2 bar, so it must equal the kernel t1_low at the signal
print("reconstructed running first-test low matches kernel t1_low at all", len(sb), "signal bars:",
      np.allclose(t1[sb], t1_at_sig, equal_nan=False))

prev_state = np.r_[0, st[:-1]]
box_switch = np.r_[False, sbox[1:] != sbox[:-1]]
active = np.isin(prev_state, (2, 3, 4)) & ~np.isnan(t1)
pierce_t1 = l <= t1 - tick
pierce_band = l < zlo
close_back = c >= np.fmax(val, t1)
m = active & (pierce_t1 | pierce_band) & close_back
print("\nIS bars (prior state TOUCH1/DEPARTED/TOUCH2) with an intrabar pierce of the first-test low (>=1 tick) or of the band"
      " low and a close back at or above max(VAL, first-test low):", int(m.sum()))
res = pd.crosstab(pd.Series(prev_state[m], name="state_before"), pd.Series(st[m], name="state_after"))
print(res)
print("  of which went to INVALID (5):", int((st[m] == 5).sum()), "; breakdown flag set on any of them:", int(brk[m].sum()))
went_armed = m & (st == 1)
print("  went to ARMED (1) on such a bar:", int(went_armed.sum()), "-> explained by a box switch at that bar:",
      int((went_armed & box_switch).sum()), "; otherwise only by a raw signal (not broken down here)")
print("  any transition to ARMED on such a bar without box switch and without raw signal:", int((went_armed & ~box_switch & ~sraw).sum()))

# all transitions into INVALID from an active state
into5 = np.isin(prev_state, (1, 2, 3, 4)) & (st == 5)
why = np.where(brk & (c < zlo), "close-based breakdown bar", np.where(box_switch, "box switch (memory block)", "UNEXPLAINED"))
print("\ntransitions into INVALID from ARMED/TOUCH1/DEPARTED/TOUCH2:", int(into5.sum()))
print(pd.Series(why[into5]).value_counts().to_dict())
print("breakdown bars with close >= VAL (should be 0):", int((brk & (c >= val)).sum()), "; with close >= band low:", int((brk & (c >= zlo)).sum()))

# merges that lowered the first-test low (state 3 -> 2 with low < running t1)
merge = (prev_state == 3) & (st == 2) & ~box_switch
print("\nmerges (DEPARTED -> TOUCH1 without separation):", int(merge.sum()), "; of which the merged touch set a new lower first-test low:",
      int((merge & (l < t1)).sum()))

# state occupancy and breakdown cancellations of active sequences
occ = pd.Series(st).map({0: "NO_BOX", 1: "ARMED", 2: "TOUCH1", 3: "DEPARTED", 4: "TOUCH2", 5: "INVALID", 6: "EXHAUSTED"}).value_counts()
print("\nIS bar occupancy by state:", (occ / n).round(4).to_dict())
canc = np.isin(prev_state, (2, 3, 4)) & (st == 5)
print("active sequences (TOUCH1/DEPARTED/TOUCH2) ended by INVALID:", int(canc.sum()), "(close-based breakdown bar:",
      int((canc & brk).sum()), ", box switch memory block:", int((canc & ~brk & box_switch).sum()), ")")
reset = np.isin(prev_state, (2, 3, 4)) & (st == 1) & box_switch & ~sraw
print("active sequences reset to ARMED by a non-carry box switch:", int(reset.sum()))

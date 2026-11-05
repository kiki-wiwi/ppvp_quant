"""s01: independent pure-Python re-implementation of the 1H state machine, written from the docstring of
strategy_state_machine.py (lists instead of swap-remove arrays), with per-sequence diagnostics.
Compares every per-bar output with the production numba kernel on the BASELINE IS run (no variant is run).

Outputs: s01_signal_diag.csv (one row per raw signal), s01_events.json (counts), stdout comparison.
"""
import json
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/ppvp_quant")
import numpy as np
import pandas as pd

from config import BASELINE_STRATEGY as CFG, IS_END
from ppvp_engine import map_boxes_to_1h, to_ns
from strategy_state_machine import indicator_arrays

OUT = Path(__file__).resolve().parent
D = pickle.load(open(OUT / "baseline_out.pkl", "rb"))
df1h, boxes, sig, trades = D["df1h"], D["boxes"], D["signals"], D["trades"]
assert df1h.index.max() < IS_END

ind = indicator_arrays(df1h, CFG)
o, h, l, c, v = ind["o"], ind["h"], ind["l"], ind["c"], ind["v"]
atr, vsma, pin, eng = ind["atr_prev"], ind["vsma_prev"], ind["pin"], ind["eng"]
open_ns = to_ns(df1h.index)
HOUR = 3600 * 10**9
box_idx = map_boxes_to_1h(df1h, boxes)
bval = boxes["VAL"].to_numpy(float)
bvah = boxes["VAH"].to_numpy(float)
bws = to_ns(boxes["window_start"])
k1, k2 = CFG.zone_k1, CFG.zone_k2
MINB, MINBOUNCE = CFG.min_bars_between_touches, CFG.min_bounce_atr
BRK_BODY, BRK_VOL, SLBUF = CFG.break_body_atr, CFG.break_vol_mult, CFG.sl_buffer_atr
assert CFG.carry_touches_on_box_update and CFG.roll_failed_second_touch and not CFG.reject_broken_at_activation
assert CFG.max_pattern_lag_bars is None and CFG.sl_anchor == "test"

n = len(df1h)
NAN = float("nan")


class Seq:
    """test sequence bookkeeping + diagnostics (diagnostics never feed back into decisions)."""

    def __init__(self):
        self.t1_low = NAN; self.t1_bar = -1; self.t1_last = -1; self.max_high = NAN
        self.t2_low = NAN; self.t2_bar = -1; self.t2_start = -1; self.seq_val = NAN; self.t2_val = NAN
        # diagnostics
        self.start_bar = -1; self.start_box = -1; self.start_at_activation = False
        self.start_prev_close_vs_zh = NAN; self.start_prev_low_above_zh = None
        self.orig_t1_low = NAN; self.orig_t1_bar = -1
        self.tests = []          # list of dicts: kind (t1/merge/t2/roll), bar, low
        self.n_merge = 0; self.n_roll = 0; self.n_carry = 0
        self.dep_bars = []       # (d0) departure bars
        self.accept = None       # diagnostics at the final second-test acceptance


seq = Seq()
st, cur = 0, -1
hv = []          # [val, t]
nd = []          # [val, t, claimed]
state = np.zeros(n, np.int8); sigr = np.zeros(n, bool); slr = np.full(n, np.nan); tpr = np.full(n, np.nan)
zlo = np.full(n, np.nan); zhi = np.full(n, np.nan); needr = np.zeros(n, bool); sboxr = np.full(n, -1)
t1lr = np.full(n, np.nan); t2lr = np.full(n, np.nan); t1br = np.full(n, -1); t2br = np.full(n, -1); t2sr = np.full(n, -1)
brkr = np.zeros(n, bool); patr = np.zeros(n, np.int8)
box_first_bar = {}
diag = []
ev = {"box_switch": 0, "carry_keep": 0, "carry_keep_in_state": {2: 0, 3: 0, 4: 0}, "reset_mid_sequence": 0,
      "hv_block": 0, "breakdown_active": 0, "breakdown_inv": 0, "merge": 0, "roll": 0, "t2_accept": 0,
      "t2_accept_by": {"bounce_only": 0, "bars_only": 0, "both": 0}, "activation_req": 0, "signal_req": 0,
      "t1_start": 0, "t1_start_at_activation_in_band": 0}
need_above = True
for i in range(n):
    b = int(box_idx[i]); a = float(atr[i])
    if b != cur:
        ev["box_switch"] += 1
        if b < 0:
            st = 0
        else:
            box_first_bar.setdefault(b, i)
            ov = (k1 + k2) * a
            hv = [m for m in hv if not (bws[b] >= m[1])]
            hv_block = (not math.isnan(a)) and any(abs(bval[b] - m[0]) <= ov for m in hv)
            keep = ((not hv_block) and cur >= 0 and st in (2, 3, 4) and not math.isnan(a)
                    and abs(bval[b] - seq.seq_val) <= ov)
            if keep:
                ev["carry_keep"] += 1; ev["carry_keep_in_state"][st] += 1; seq.n_carry += 1
            else:
                if st in (2, 3, 4):
                    ev["reset_mid_sequence"] += 1
                seq = Seq()
                for r in nd:
                    r[2] = (not math.isnan(a)) and abs(bval[b] - r[0]) <= ov
                need_above = any(r[2] for r in nd)
            nd = [r for r in nd if not ((not r[2]) and bws[b] >= r[1])]
            if hv_block:
                st = 5; ev["hv_block"] += 1
            elif not keep:
                st = 1
                zl_new = bval[b] - k1 * a
                below = not (i > 0 and c[i - 1] >= zl_new)
                if below and not need_above:
                    nd.append([bval[b], int(open_ns[i]), True]); need_above = True; ev["activation_req"] += 1
        cur = b
    sboxr[i] = cur
    if st == 0 or math.isnan(a):
        state[i] = st; needr[i] = need_above; continue
    zl = bval[cur] - k1 * a; zh = bval[cur] + k2 * a
    zlo[i] = zl; zhi[i] = zh
    if nd:
        nd = [r for r in nd if not (l[i] > (zh if r[2] else r[0] + k2 * a))]
        need_above = any(r[2] for r in nd)
    is_brk = (c[i] < o[i]) and (c[i] < zl) and (o[i] - c[i] >= BRK_BODY * a) and (v[i] >= BRK_VOL * vsma[i])
    brkr[i] = is_brk
    if st in (5, 6):
        if is_brk:
            ev["breakdown_inv"] += 1
            same = False
            if st == 5:
                for m in hv:
                    if abs(bval[cur] - m[0]) <= (k1 + k2) * a:
                        same = True
                        if c[i] < m[0] - k1 * a:
                            m[1] = int(open_ns[i]) + HOUR
            if not same:
                hv.append([bval[cur], int(open_ns[i]) + HOUR]); st = 5
        state[i] = st; needr[i] = need_above; continue
    if is_brk:
        ev["breakdown_active"] += 1
        st = 5; hv.append([bval[cur], int(open_ns[i]) + HOUR]); seq = Seq()
        state[i] = st; needr[i] = need_above; continue
    touched = l[i] <= zh
    if st == 1:
        if (not need_above) and touched:
            st = 2; ev["t1_start"] += 1
            seq = Seq()
            seq.t1_low = l[i]; seq.t1_bar = i; seq.t1_last = i; seq.seq_val = bval[cur]
            seq.start_bar = i; seq.start_box = cur
            seq.start_at_activation = (box_first_bar.get(cur) == i)
            seq.start_prev_close_vs_zh = (c[i - 1] - zh) / a
            seq.start_prev_low_above_zh = bool(l[i - 1] > zh)
            if seq.start_at_activation and c[i - 1] <= zh:
                ev["t1_start_at_activation_in_band"] += 1
            seq.tests.append({"kind": "t1", "bar": i})
    elif st == 2:
        if touched:
            if l[i] < seq.t1_low:
                seq.t1_low = l[i]; seq.t1_bar = i
            seq.t1_last = i
        else:
            st = 3; seq.max_high = h[i]; seq.dep_bars.append(i)
    elif st == 3:
        if touched:
            by_bounce = (seq.max_high - zh) >= MINBOUNCE * a
            by_bars = (i - seq.t1_last) >= MINB
            if by_bounce or by_bars:
                st = 4; ev["t2_accept"] += 1
                ev["t2_accept_by"]["both" if (by_bounce and by_bars) else "bounce_only" if by_bounce else "bars_only"] += 1
                seq.t2_low = l[i]; seq.t2_bar = i; seq.t2_start = i; seq.t2_val = bval[cur]
                d0 = seq.dep_bars[-1]
                seq.accept = {"t2_start": i, "d0": d0, "t1_last": seq.t1_last, "gap": i - seq.t1_last,
                              "max_high": seq.max_high, "zh_t2": zh, "atr_t2": a,
                              "bounce_atr": (seq.max_high - zh) / a,
                              "zh_d0": zhi[d0], "atr_d0": atr[d0],
                              "bounce_vs_d0_band_atr": (seq.max_high - zhi[d0]) / a,
                              "box_d0": int(sboxr[d0]), "box_t2": cur,
                              "by_bounce": bool(by_bounce), "by_bars": bool(by_bars),
                              "t2_low_vs_prev_bar_band": bool(l[i] <= zhi[i - 1]),
                              "d0_low_vs_prev_bar_band": bool(l[d0] > zhi[d0 - 1]) if d0 - 1 >= 0 and not np.isnan(zhi[d0 - 1]) else None}
                seq.tests.append({"kind": "t2", "bar": i})
            else:
                st = 2; ev["merge"] += 1; seq.n_merge += 1
                if l[i] < seq.t1_low:
                    seq.t1_low = l[i]; seq.t1_bar = i
                seq.t1_last = i
                seq.tests.append({"kind": "merge", "bar": i})
        else:
            if h[i] > seq.max_high:
                seq.max_high = h[i]
    elif st == 4:
        if touched:
            if l[i] < seq.t2_low:
                seq.t2_low = l[i]; seq.t2_bar = i
        else:
            st = 3; ev["roll"] += 1; seq.n_roll += 1
            if np.isnan(seq.orig_t1_low):
                seq.orig_t1_low = seq.t1_low; seq.orig_t1_bar = seq.t1_bar
            seq.orig_all_low = min(getattr(seq, "orig_all_low", np.inf), seq.t1_low)
            seq.t1_low = seq.t2_low; seq.t1_bar = seq.t2_bar; seq.t1_last = i - 1; seq.seq_val = seq.t2_val
            seq.max_high = h[i]; seq.t2_low = NAN; seq.t2_bar = -1; seq.t2_start = -1; seq.t2_val = NAN
            seq.dep_bars.append(i)
            seq.tests.append({"kind": "roll", "bar": i})
    if st == 4 and touched and c[i] >= zl:
        p = (1 if pin[i] else 0) | (2 if eng[i] else 0)
        if p > 0:
            lowest = min(seq.t1_low, seq.t2_low)
            eng_prev_used = False
            if (p & 2) and i > 0 and l[i - 1] < lowest:
                lowest = l[i - 1]; eng_prev_used = True
            sigr[i] = True; patr[i] = p; slr[i] = lowest - SLBUF * a; tpr[i] = bvah[cur]
            t1lr[i] = seq.t1_low; t1br[i] = seq.t1_bar; t2lr[i] = seq.t2_low; t2br[i] = seq.t2_bar; t2sr[i] = seq.t2_start
            # test-2 bars entirely below the band (high < zone low at that bar)
            t2bars = np.arange(seq.t2_start, i + 1)
            below_band = int(np.sum(h[t2bars] < zlo[t2bars]))
            # test-1 bars: from start_bar to t1_last (including merged touches and departures in between)
            rec = {"signal_bar": i, "signal_time": df1h.index[i], "box_signal": cur,
                   "box_seq_start": seq.start_box, "seq_start_bar": seq.start_bar,
                   "seq_start_at_activation": seq.start_at_activation,
                   "seq_start_prev_close_vs_zh_atr": seq.start_prev_close_vs_zh,
                   "seq_start_prev_low_above_zh": seq.start_prev_low_above_zh,
                   "n_merge": seq.n_merge, "n_roll": seq.n_roll, "n_carry": seq.n_carry,
                   "t1_low": seq.t1_low, "t2_low": seq.t2_low, "orig_t1_low": seq.orig_t1_low,
                   "orig_all_low": getattr(seq, "orig_all_low", np.nan),
                   "pattern": p, "eng_prev_used": eng_prev_used, "sl": slr[i], "tp": tpr[i],
                   "atr": a, "zl": zl, "zh": zh, "VAL": bval[cur], "VAH": bvah[cur], "close": c[i],
                   "low": l[i], "high": h[i], "open": o[i],
                   "lag_from_t2_start": i - seq.t2_start, "t2_bars": len(t2bars), "t2_bars_below_band": below_band,
                   "close_below_VAL": bool(c[i] < bval[cur]), "close_below_t1_low": bool(c[i] < seq.t1_low),
                   "val_at_seq_start": bval[seq.start_box] if seq.start_box >= 0 else np.nan,
                   "vah_at_seq_start": bvah[seq.start_box] if seq.start_box >= 0 else np.nan,
                   "seq_val_at_signal": seq.seq_val}
            rec.update({f"acc_{k}": vv for k, vv in seq.accept.items()})
            diag.append(rec)
            st = 1
            nd.append([bval[cur], int(open_ns[i]) + HOUR, True]); need_above = True; ev["signal_req"] += 1
            seq = Seq()
    state[i] = st; needr[i] = need_above

# ---- compare with production kernel output ----
cmp = {
    "state": np.array_equal(state, sig["state"].to_numpy()),
    "sig_raw": np.array_equal(sigr, sig["sig_raw"].to_numpy()),
    "pattern": np.array_equal(patr, sig["pattern"].to_numpy()),
    "breakdown": np.array_equal(brkr, sig["breakdown"].to_numpy()),
    "need_above": np.array_equal(needr, sig["need_above"].to_numpy()),
    "box_id": np.array_equal(sboxr, sig["box_id"].to_numpy()),
    "zone_lo": np.array_equal(zlo, sig["zone_lo"].to_numpy(), equal_nan=True),
    "zone_hi": np.array_equal(zhi, sig["zone_hi"].to_numpy(), equal_nan=True),
    "sl": np.array_equal(slr, sig["sl"].to_numpy(), equal_nan=True),
    "tp": np.array_equal(tpr, sig["tp"].to_numpy(), equal_nan=True),
    "t1_low": np.array_equal(t1lr, sig["t1_low"].to_numpy(), equal_nan=True),
    "t2_low": np.array_equal(t2lr, sig["t2_low"].to_numpy(), equal_nan=True),
    "t1_bar": np.array_equal(t1br, sig["t1_bar"].to_numpy()),
    "t2_bar": np.array_equal(t2br, sig["t2_bar"].to_numpy()),
    "t2_start": np.array_equal(t2sr, sig["t2_start"].to_numpy()),
}
print("bit-identical to production kernel per output:", cmp)
print("ALL IDENTICAL:", all(cmp.values()))
dg = pd.DataFrame(diag)
# attach trade status from the committed ledger (status only)
led = pd.read_csv("/home/user/ppvp_quant/reports/audit_round/signal_opportunities_B_original.csv")
assert np.array_equal(led["signal_bar"].to_numpy(), dg["signal_bar"].to_numpy())
dg["status"] = led["status"].to_numpy()
dg["ledger_qualified_by"] = led["second_test_qualified_by"].to_numpy()
dg.to_csv(OUT / "s01_signal_diag.csv", index=False)
json.dump(ev, open(OUT / "s01_events.json", "w"), indent=1, default=int)
print("events:", json.dumps(ev, default=int))
print("raw signals", len(dg), "filled", int((dg.status == "filled").sum()))

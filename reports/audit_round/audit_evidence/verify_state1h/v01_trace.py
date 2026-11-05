"""v01: instrumented pure-Python copy of strategy_state_machine._run_state_machine (typed from the source),
with per-bar event logging, chain ids and requirement bookkeeping. BASELINE, IS only (objects from v00).
Checks all 15 outputs bit-for-bit against the production signals frame, then pickles the trace."""
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/ppvp_quant")
import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent
O = pickle.load(open(D / "v00_objs.pkl", "rb"))
df1h, sig, boxes, args = O["df1h"], O["signals"], O["boxes"], O["args"]
(o, h, l, c, v, atr, vsma, open_ns, box_idx, box_val, box_vah, box_wstart, pin, eng,
 k1, k2, min_bars, min_bounce, carry, roll, brk_body, brk_vol, sl_buf, reject_broken, max_lag, sl_pattern) = args
assert (k1, k2, min_bars, min_bounce, carry, roll, reject_broken, max_lag, sl_pattern) == (0.5, 0.5, 6, 1.0, True, True, False, -1, False)
n = len(o)
HOUR = 3600 * 10**9
isnan = math.isnan

state = np.zeros(n, np.int8); zlo = np.full(n, np.nan); zhi = np.full(n, np.nan)
brk = np.zeros(n, bool); sigr = np.zeros(n, bool); pat = np.zeros(n, np.int8)
sl = np.full(n, np.nan); tp = np.full(n, np.nan)
o_t1low = np.full(n, np.nan); o_t1bar = np.full(n, -1, np.int64); o_t2low = np.full(n, np.nan)
o_t2bar = np.full(n, -1, np.int64); o_t2start = np.full(n, -1, np.int64); o_box = np.full(n, -1, np.int64)
o_need = np.zeros(n, bool)
# instrumentation
pre_state = np.zeros(n, np.int8)           # state at the start of bar i (before any processing)
post_switch_state = np.full(n, -1, np.int8)  # state right after the box-switch block (if a switch happened)
switch_kind = np.array([""] * n, dtype=object)
ev = [[] for _ in range(n)]               # list of event strings per bar
chain_id = np.full(n, -1, np.int64)       # chain the bar belongs to (after processing), signal bar included
t1last_at = np.full(n, -1, np.int64)
maxhigh_at = np.full(n, np.nan)
seqval_at = np.full(n, np.nan)
reqs = []                                 # requirement records
st_mid = np.zeros(n, np.int8); t1low_mid = np.full(n, np.nan)

st = 0; cur = -1
t1_low = np.nan; t1_bar = -1; t1_last = -1; max_high = np.nan
t2_low = np.nan; t2_bar = -1; t2_start = -1; seq_val = np.nan; t2_val = np.nan
hv_vals, hv_ts = [], []
nd = []          # list of dicts: val, t, cl, id
need_above = True
cid = -1; next_cid = 0
chains = {}

for i in range(n):
    pre_state[i] = st
    b = int(box_idx[i]); a = float(atr[i])
    if b != cur:
        if b < 0:
            st = 0
            switch_kind[i] = "to_nobox"
            cid = -1
        else:
            ov = (k1 + k2) * a
            # expire hv memories (order-insensitive filter)
            keep_hv = [(vv, tt) for vv, tt in zip(hv_vals, hv_ts) if not (box_wstart[b] >= tt)]
            hv_vals = [x[0] for x in keep_hv]; hv_ts = [x[1] for x in keep_hv]
            hv_block = False
            if not isnan(a):
                for vv in hv_vals:
                    if abs(box_val[b] - vv) <= ov:
                        hv_block = True
                        break
            keep = False
            if (not hv_block) and carry and cur >= 0 and st in (2, 3, 4) and not isnan(a):
                if abs(box_val[b] - seq_val) <= ov:
                    keep = True
            if hv_block or not keep:
                if st in (2, 3, 4):
                    ev[i].append(f"reset_mid_seq_from_{st}" + ("_hvblock" if hv_block else ""))
                t1_low = np.nan; t1_bar = -1; t1_last = -1; max_high = np.nan
                t2_low = np.nan; t2_bar = -1; t2_start = -1; seq_val = np.nan; t2_val = np.nan
                need_above = False
                for r in nd:
                    r["cl"] = (not isnan(a)) and abs(box_val[b] - r["val"]) <= ov
                    if r["cl"]:
                        need_above = True
                        reqs[r["id"]]["claims"].append((i, b))
                cid = -1
            else:
                ev[i].append(f"carry_keep_in_{st}")
                chains[cid]["carries"].append((i, cur, b))
            # unclaimed requirements expire on fresh structure
            survivors = []
            for r in nd:
                if (not r["cl"]) and box_wstart[b] >= r["t"]:
                    reqs[r["id"]]["fate"] = ("expired_fresh_structure", i)
                else:
                    survivors.append(r)
            nd = survivors
            if hv_block:
                st = 5
                switch_kind[i] = "hv_block"
            elif not keep:
                st = 1
                switch_kind[i] = "arm"
                zl_new = box_val[b] - k1 * a
                below = not (i > 0 and c[i - 1] >= zl_new)
                if below and not need_above:
                    rid = len(reqs)
                    reqs.append({"id": rid, "kind": "activation", "bar": i, "val": box_val[b], "box": b, "claims": [], "fate": None})
                    nd.append({"val": box_val[b], "t": int(open_ns[i]), "cl": True, "id": rid})
                    need_above = True
                if reject_broken and i > 0 and not isnan(a) and c[i - 1] < zl_new:
                    st = 5
            else:
                switch_kind[i] = "carry"
            post_switch_state[i] = st
        cur = b
    o_box[i] = cur
    st_mid[i] = st; t1low_mid[i] = t1_low
    if st == 0 or isnan(a):
        state[i] = st; o_need[i] = need_above
        continue
    zl = box_val[cur] - k1 * a
    zh = box_val[cur] + k2 * a
    zlo[i] = zl; zhi[i] = zh
    if len(nd) > 0:
        survivors = []
        for r in nd:
            thr = zh if r["cl"] else r["val"] + k2 * a
            if l[i] > thr:
                reqs[r["id"]]["fate"] = ("cleared_claimed" if r["cl"] else "cleared_unclaimed", i)
            else:
                survivors.append(r)
        nd = survivors
        need_above = any(r["cl"] for r in nd)
    is_brk = (c[i] < o[i]) and (c[i] < zl) and (o[i] - c[i] >= brk_body * a) and (v[i] >= brk_vol * vsma[i])
    brk[i] = is_brk
    if st == 5 or st == 6:
        if is_brk:
            same = False
            if st == 5:
                for e in range(len(hv_vals)):
                    if abs(box_val[cur] - hv_vals[e]) <= (k1 + k2) * a:
                        same = True
                        if c[i] < hv_vals[e] - k1 * a:
                            hv_ts[e] = int(open_ns[i]) + HOUR
            if not same:
                hv_vals.append(box_val[cur]); hv_ts.append(int(open_ns[i]) + HOUR)
                st = 5
        state[i] = st; o_need[i] = need_above
        continue
    if is_brk:
        if st in (2, 3, 4):
            ev[i].append(f"brk_ends_seq_from_{st}")
        ev[i].append(f"brk_into_invalid_from_{st}")
        st = 5
        hv_vals.append(box_val[cur]); hv_ts.append(int(open_ns[i]) + HOUR)
        cid = -1
        state[i] = st; o_need[i] = need_above
        continue
    touched = l[i] <= zh
    if st == 1:
        if need_above:
            pass
        elif touched:
            st = 2
            t1_low = l[i]; t1_bar = i; t1_last = i; seq_val = box_val[cur]
            cid = next_cid; next_cid += 1
            chains[cid] = {"start": i, "carries": [], "events": [], "start_box": cur,
                           "start_at_switch": switch_kind[i] == "arm"}
            ev[i].append("t1_start")
    elif st == 2:
        if touched:
            if l[i] < t1_low:
                t1_low = l[i]; t1_bar = i
            t1_last = i
        else:
            st = 3
            max_high = h[i]
            ev[i].append("depart")
    elif st == 3:
        if touched:
            bounce_ok = (max_high - zh >= min_bounce * a)
            bars_ok = (i - t1_last >= min_bars)
            if bounce_ok or bars_ok:
                st = 4
                t2_low = l[i]; t2_bar = i; t2_start = i; t2_val = box_val[cur]
                ev[i].append("t2_accept_" + ("both" if bounce_ok and bars_ok else "bounce_only" if bounce_ok else "bars_only"))
                chains[cid]["events"].append(("t2_accept", i, (max_high - zh) / a, i - t1_last))
            else:
                st = 2
                lower = l[i] < t1_low
                if lower:
                    t1_low = l[i]; t1_bar = i
                t1_last = i
                ev[i].append("merge" + ("_lower" if lower else ""))
                chains[cid]["events"].append(("merge", i, lower, (max_high - zh) / a, i - t1_last))
        else:
            if h[i] > max_high:
                max_high = h[i]
    elif st == 4:
        if touched:
            if l[i] < t2_low:
                t2_low = l[i]; t2_bar = i
        else:
            if roll:
                st = 3
                chains[cid]["events"].append(("roll", i, t1_low, t2_low))
                t1_low = t2_low; t1_bar = t2_bar; t1_last = i - 1; seq_val = t2_val; max_high = h[i]
                t2_low = np.nan; t2_bar = -1; t2_start = -1; t2_val = np.nan
                ev[i].append("roll")
            else:
                st = 6
    if st == 4 and touched and c[i] >= zl and (max_lag < 0 or i - t2_start <= max_lag):
        p = 0
        if pin[i]:
            p |= 1
        if eng[i]:
            p |= 2
        if p > 0:
            lowest = min(t1_low, l[i]) if sl_pattern else min(t1_low, t2_low)
            eng_used = False
            if (p & 2) and i > 0 and l[i - 1] < lowest:
                lowest = l[i - 1]; eng_used = True
            sigr[i] = True; pat[i] = p; sl[i] = lowest - sl_buf * a; tp[i] = box_vah[cur]
            o_t1low[i] = t1_low; o_t1bar[i] = t1_bar; o_t2low[i] = t2_low; o_t2bar[i] = t2_bar; o_t2start[i] = t2_start
            chains[cid]["signal"] = i
            chains[cid]["eng_prev_used"] = eng_used
            chains[cid]["t1_last_at_signal"] = t1_last
            ev[i].append("signal")
            chain_id[i] = cid
            st = 1
            rid = len(reqs)
            reqs.append({"id": rid, "kind": "signal", "bar": i, "val": box_val[cur], "box": cur, "claims": [], "fate": None})
            nd.append({"val": box_val[cur], "t": int(open_ns[i]) + HOUR, "cl": True, "id": rid})
            need_above = True
            t1_low = np.nan; t1_bar = -1; t1_last = -1; max_high = np.nan
            t2_low = np.nan; t2_bar = -1; t2_start = -1; seq_val = np.nan; t2_val = np.nan
            cid = -1
    if st in (2, 3, 4):
        chain_id[i] = cid
    t1last_at[i] = t1_last; maxhigh_at[i] = max_high; seqval_at[i] = seq_val
    state[i] = st; o_need[i] = need_above

# ---------------- bit-identity vs production ----------------
def same(a_, b_):
    a_ = np.asarray(a_); b_ = np.asarray(b_)
    if a_.dtype.kind == "f" or b_.dtype.kind == "f":
        a_ = a_.astype(float); b_ = b_.astype(float)
        return bool(np.array_equal(a_, b_, equal_nan=True))
    return bool(np.array_equal(a_, b_))

prod = {"state": sig["state"], "zone_lo": sig["zone_lo"], "zone_hi": sig["zone_hi"], "breakdown": sig["breakdown"],
        "sig_raw": sig["sig_raw"], "pattern": sig["pattern"], "sl": sig["sl"], "tp": sig["tp"], "t1_low": sig["t1_low"],
        "t1_bar": sig["t1_bar"], "t2_low": sig["t2_low"], "t2_bar": sig["t2_bar"], "t2_start": sig["t2_start"],
        "box_id": sig["box_id"], "need_above": sig["need_above"]}
mine = {"state": state, "zone_lo": zlo, "zone_hi": zhi, "breakdown": brk, "sig_raw": sigr, "pattern": pat, "sl": sl,
        "tp": tp, "t1_low": o_t1low, "t1_bar": o_t1bar, "t2_low": o_t2low, "t2_bar": o_t2bar, "t2_start": o_t2start,
        "box_id": o_box, "need_above": o_need}
# production fills t1_bar etc. only on signal bars (others -1 / NaN); compare everywhere
res = {k: same(mine[k], prod[k].to_numpy()) for k in prod}
print("bit-identical per output:", res)
print("ALL IDENTICAL:", all(res.values()))
from collections import Counter
cnt = Counter(e for lst in ev for e in lst)
print("event counts:", dict(sorted(cnt.items())))
print("switch kinds:", dict(Counter(switch_kind[switch_kind != ""])))
print("signals", int(sigr.sum()), "chains with signal", sum(1 for ch in chains.values() if "signal" in ch))
with open(D / "v01_trace.pkl", "wb") as f:
    pickle.dump({"state": state, "pre_state": pre_state, "post_switch_state": post_switch_state, "switch_kind": switch_kind,
                 "ev": ev, "chain_id": chain_id, "chains": chains, "reqs": reqs, "t1last_at": t1last_at,
                 "maxhigh_at": maxhigh_at, "seqval_at": seqval_at, "zlo": zlo, "zhi": zhi, "box": o_box,
                 "sig": sigr, "pat": pat, "sl": sl, "tp": tp, "t1low": o_t1low, "t1bar": o_t1bar, "t2low": o_t2low,
                 "t2bar": o_t2bar, "t2start": o_t2start, "need": o_need, "brk": brk, "st_mid": st_mid, "t1low_mid": t1low_mid}, f)

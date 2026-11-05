"""s03: prefix-truncation checks of every IS box around its available_time (engine build_boxes on IS prefixes).

For box k with available time A_k (= close of 4H bar b+R):
  cut "before": 1H bars with open < A_k - 1h  (4H bar b+R lacks its last 1H bar -> must be dropped, box k absent)
  cut "at":     1H bars with open < A_k       (4H bar b+R just closed -> box k present)
In both cases every box present must equal the full-IS box with the same id bit for bit (all columns).
Also: random mid-4H cuts; and 1H mapping boundary (bar opening at A_k - 1h uses box k-1, bar opening at A_k uses box k).
"""
import sys, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/ppvp_quant")
HERE = Path(__file__).resolve().parent
from ppvp_engine import build_boxes, build_4h, map_boxes_to_1h
from config import BASELINE_STRATEGY as S

d = pickle.load(open(HERE / "is_objects.pkl", "rb"))
df1h, boxes, sig = d["df1h"], d["boxes"], d["sig"]
H = pd.Timedelta(hours=1)
avail = pd.DatetimeIndex(boxes["available_time"])


def same_prefix(bt, k_expected):
    if len(bt) != k_expected:
        return False, f"n={len(bt)} expected {k_expected}"
    ref = boxes.iloc[:k_expected]
    for c in boxes.columns:
        a, b = bt[c].to_numpy(), ref[c].to_numpy()
        if a.dtype != b.dtype or not np.array_equal(a, b):
            return False, f"col {c} differs"
    return True, ""


t0 = time.time()
bad = []
partial_dropped = 0
for k in range(len(boxes)):
    A = avail[k]
    cut_b = df1h[df1h.index < A - H]
    cut_a = df1h[df1h.index < A]
    bb, d4b = build_boxes(cut_b, S)
    ba, d4a = build_boxes(cut_a, S)
    ok1, m1 = same_prefix(bb, k)
    ok2, m2 = same_prefix(ba, k + 1)
    # the 4H bar b+R (open = A - 4h) must be absent in the "before" cut and present in the "at" cut
    last4_b, last4_a = d4b.index[-1], d4a.index[-1]
    if last4_b == A - 8 * H and last4_a == A - 4 * H:
        partial_dropped += 1
    if not (ok1 and ok2):
        bad.append((k, m1, m2))
print(f"box prefix cuts: {len(boxes)} boxes x 2 cuts, failures: {len(bad)}  ({time.time()-t0:.0f}s)")
print("partial 4H bar dropped in every 'before' cut and present in every 'at' cut:", partial_dropped == len(boxes))
if bad:
    print(bad[:10])

# random mid-4H cuts
rng = np.random.default_rng(1105)
cuts = rng.choice(np.arange(200, len(df1h)), size=60, replace=False)
bad2 = 0
for j in cuts:
    sub = df1h.iloc[: j + 1]
    bt, d4 = build_boxes(sub, S)
    last_close = sub.index[-1] + H
    k_exp = int((avail <= last_close).sum())          # boxes known by the close of the last bar
    ok, m = same_prefix(bt, k_exp)
    full4 = build_4h(df1h)
    ok4b = bool(((d4.index + 4 * H) <= last_close).all()) and len(d4) == int(((full4.index + 4 * H) <= last_close).sum())
    if not (ok and ok4b):
        bad2 += 1
print("random cuts:", len(cuts), "failures:", bad2)

# 1H mapping boundary
bid = sig["box_id"].to_numpy()
pos = {t: i for i, t in enumerate(df1h.index)}
viol = 0
for k in range(len(boxes)):
    A = avail[k]
    i_at = pos.get(A)
    i_bef = pos.get(A - H)
    if i_at is None:
        continue          # last boxes may become available at IS_END
    if bid[i_at] != k or (i_bef is not None and bid[i_bef] != k - 1):
        viol += 1
n_eval = sum(1 for k in range(len(boxes)) if avail[k] in pos)
print(f"mapping boundary: {n_eval} boxes checked, violations {viol}; boxes whose available_time has no 1H bar in IS:",
      [str(a) for a in avail if a not in pos])
m = map_boxes_to_1h(df1h, boxes)
print("state-machine box_id == map_boxes_to_1h everywhere:", bool((m == bid).all()))
# no 1H bar uses a box whose available_time > its open time
use = bid >= 0
print("any bar using a box with available_time > open_time:", bool((pd.DatetimeIndex(avail[bid[use]]) > df1h.index[use]).any()))

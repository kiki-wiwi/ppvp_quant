# (1) IS-only prefix truncation: data -> 4H -> boxes -> signals on IS prefixes must equal the full-IS run on the prefix.
# (2) Structural effect of the unflagged partial hour 2021-03-02 01:00 (64 BTC, 457 trades) on signals: keep (baseline)
#     vs treat as a missing maintenance hour. Structural outputs only (no performance of any alternative is computed).
import numpy as np
import pandas as pd
from ppvp_audit.common import is_run, load_is_market
from config import BASELINE_STRATEGY as S
from ppvp_engine import build_boxes
from strategy_state_machine import generate_signals

out = is_run()
df1h_full, sig_full, boxes_full = out["df1h"], out["signals"], out["boxes"]

def eq_frame(a, b):
    bad = []
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        ok = np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f" else np.array_equal(x, y)
        if not ok:
            bad.append(c)
    return bad

cuts = ["2020-01-04 17:00", "2020-03-12 13:00", "2021-03-02 02:00", "2021-05-19 15:00", "2022-06-18 02:00",
        "2022-11-09 23:00", "2023-11-10 14:00", "2023-12-31 22:00"]
for cstr in cuts:
    cut = pd.Timestamp(cstr, tz="UTC")
    d = df1h_full[df1h_full.index < cut]
    bx, d4 = build_boxes(d, S)
    sg = generate_signals(d, S, bx, d4, None)
    ref = sig_full.loc[sg.index]
    nb = len(bx)
    box_bad = eq_frame(bx.reset_index(drop=True), boxes_full.iloc[:nb].reset_index(drop=True))
    later_avail = int((pd.DatetimeIndex(boxes_full["available_time"]) <= cut).sum())
    print("cut %s: prefix bars %d, boxes %d (full-run boxes available <= cut: %d), box cols differing %s, signal cols differing %s"
          % (cstr, len(d), nb, later_avail, box_bad, eq_frame(sg, ref)))

# (2) thin hour
t = pd.Timestamp("2021-03-02 01:00", tz="UTC")
d2 = df1h_full.drop(index=t)
bx2, d42 = build_boxes(d2, S)
sg2 = generate_signals(d2, S, bx2, d42, None)
print("\nthin-hour treatment: boxes baseline %d vs dropped %d" % (len(boxes_full), len(bx2)))
d4b = out["df4h"]
print("4H 2021-03-02 00:00 baseline:", d4b.loc[pd.Timestamp("2021-03-02", tz="UTC"), ["open", "high", "low", "close", "volume", "n_bars"]].to_dict())
print("4H 2021-03-02 00:00 dropped :", d42.loc[pd.Timestamp("2021-03-02", tz="UTC"), ["open", "high", "low", "close", "volume", "n_bars"]].to_dict())
common = sg2.index
a, b = sig_full.loc[common], sg2
for c in ["sig_raw", "signal", "breakdown", "state", "box_id", "sl", "tp", "VAL", "VAH"]:
    x, y = a[c].to_numpy(), b[c].to_numpy()
    diff = ~((x == y) | (pd.isna(x) & pd.isna(y))) if x.dtype.kind == "f" else (x != y)
    print("   %-10s rows differing: %d" % (c, int(diff.sum())), ("first " + str(common[np.flatnonzero(diff)[0]]) if diff.any() else ""))
w = (common > t) & (common <= t + pd.Timedelta(hours=24))
rv = (a["vsma"] / b["vsma"] - 1)[w]
ra = (a["atr"] / b["atr"] - 1)[w]
print("   vsma rel diff over next 24h: max %.4f ; atr rel diff max %.4f" % (rv.abs().max(), ra.abs().max()))
# volume/vsma margin to the 1.5x breakdown threshold for bearish bars in the 20 bars after the thin hour
v = df1h_full["volume"]; o = df1h_full["open"]; c = df1h_full["close"]
w20 = (df1h_full.index > t) & (df1h_full.index <= t + pd.Timedelta(hours=21))
x = pd.DataFrame({"v/vsma_base": (v / sig_full["vsma"])[w20], "bear": (c < o)[w20], "below_zone": (c < sig_full["zone_lo"])[w20],
                  "body/atr": ((o - c) / sig_full["atr"])[w20], "state": sig_full["state"][w20]})
print(x.to_string())
tr = out["result"].trades
near = tr[(tr["entry_time"] > t - pd.Timedelta(days=3)) & (tr["entry_time"] < t + pd.Timedelta(days=5))]
print("trades entered within [-3d, +5d] of the thin hour:", len(near), list(near["entry_time"].astype(str)))

"""s03: concrete examples + quantification on the BASELINE IS run (109 filled trades / 120 raw signals)."""
import pickle
import sys

sys.path.insert(0, "/home/user/ppvp_quant")
import numpy as np
import pandas as pd

from config import IS_END

Dd = "/tmp/claude-0/-home-user-ppvp-quant/04a66f10-1e5e-5960-9a9f-ffb5b694e43a/scratchpad/audit_round/state1h"
P = pickle.load(open(f"{Dd}/baseline_out.pkl", "rb"))
df1h, sig, boxes, trades = P["df1h"], P["signals"], P["boxes"], P["trades"]
assert df1h.index.max() < IS_END
dg = pd.read_csv(f"{Dd}/s01_signal_diag.csv")
f = pd.read_csv(f"{Dd}/s02_filled_diag.csv")
idx = df1h.index
h, l, c, o = (df1h[k].to_numpy(float) for k in ("high", "low", "close", "open"))
zlo, zhi, atr, st = (sig[k].to_numpy() for k in ("zone_lo", "zone_hi", "atr", "state"))
sbox = sig.box_id.to_numpy()
pd.set_option("display.width", 250, "display.max_columns", 40)


def T(i):
    return str(idx[int(i)])[:16]


# trades table SL/TP equal signal SL/TP at entry bar
tr = trades.set_index("entry_bar")
eb = tr.index.to_numpy()
print("trades SL == sig.sl at entry bar:", np.array_equal(tr["sl"].to_numpy(), sig["sl"].to_numpy()[eb]),
      " TP == sig.tp:", np.array_equal(tr["tp"].to_numpy(), sig["tp"].to_numpy()[eb]))

# ---- 1. test-1 depth / below-band bars, stop distance ----
rows = []
for _, r in f.iterrows():
    s0, t1l = int(r.seq_start_bar), int(r.acc_t1_last)
    bars = np.arange(s0, t1l + 1)
    in_t1 = bars[st[bars] == 2]
    rows.append({"t1_bars_state2": len(in_t1), "t1_bars_below_band": int(np.sum(h[in_t1] < zlo[in_t1])),
                 "t1_depth_below_zl_atr": (zlo[int(r.signal_bar)] - r.t1_low) / r.atr})
x = pd.DataFrame(rows)
f = pd.concat([f, x], axis=1)
print("\n[1] filled trades whose (final) first-test bars include a bar entirely below the band:",
      int((f.t1_bars_below_band > 0).sum()))
print("    t1_low below signal-bar band low (ATR) quantiles:", f.t1_depth_below_zl_atr.quantile([0, .5, .9, 1]).round(2).to_dict())
f["stop_dist_atr"] = (f.close - f.sl) / f.atr
print("    stop distance (close-SL)/ATR quantiles:", f.stop_dist_atr.quantile([0, .1, .5, .9, 1]).round(2).to_dict())

# ---- 2. t2 bars below band example ----
e = f[f.t2_bars > 0][f.t2_bars_below_band > 0]
print("\n[2] signal with a test-2 bar entirely below band:")
print(e[["signal_time", "acc_t2_start", "lag_from_t2_start", "t2_bars", "t2_bars_below_band", "zl", "zh", "t2_low", "close", "status"]])
for _, r in e.iterrows():
    for j in range(int(r.acc_t2_start), int(r.signal_bar) + 1):
        print("   ", T(j), "state", st[j], "O/H/L/C", o[j], h[j], l[j], c[j], "band", round(zlo[j], 2), round(zhi[j], 2))

# ---- 3. close below VAL / below t1_low ----
cb = f[f.close_below_VAL | f.close_below_t1_low]
print("\n[3] filled signals closing below VAL or below the first-test low:", len(cb))
print(cb[["signal_time", "VAL", "zl", "t1_low", "t2_low", "low", "close", "pattern", "close_below_VAL", "close_below_t1_low"]].round(2).to_string())

# ---- 4. rolled sequences: earlier lower low not in SL ----
rl = f[(f.n_roll > 0) & (f.orig_all_low < np.minimum(f.t1_low, f.t2_low))].copy()
rl["sl_if_all_tests"] = rl.orig_all_low - 0.2 * rl.atr
rl["sl_gap_atr"] = (rl.sl - rl.sl_if_all_tests) / rl.atr
rl["sl_gap_pct_of_close"] = (rl.sl - rl.sl_if_all_tests) / rl.close * 100
print("\n[4] filled trades after >=1 roll:", int((f.n_roll > 0).sum()), "; with an earlier test low below min(t1_low, t2_low):", len(rl))
print(rl[["signal_time", "n_roll", "orig_all_low", "t1_low", "t2_low", "sl", "sl_gap_atr", "sl_gap_pct_of_close"]].round(3).to_string())

# ---- 5. sequences starting inside the band at box activation ----
ib = f[(f.t1_origin == "prev close inside band")]
print("\n[5] filled trades whose sequence began at a box activation bar with the prior close already inside the band:", len(ib),
      "; of which no roll (the signal's 'first test' is that in-band start):", int((ib.n_roll == 0).sum()))
ex = ib.iloc[0]
s0 = int(ex.seq_start_bar)
print("    example:", ex.signal_time, "box", int(ex.box_seq_start), "activation bar", T(s0))
for j in range(s0 - 2, s0 + 2):
    print("     ", T(j), "box", sbox[j], "state", st[j], "H/L/C", h[j], l[j], c[j], "band", None if np.isnan(zlo[j]) else (round(zlo[j], 2), round(zhi[j], 2)))
print("     new box VAL", boxes.VAL.iloc[int(ex.box_seq_start)], "ATR", atr[s0], "-> band",
      boxes.VAL.iloc[int(ex.box_seq_start)] - 0.5 * atr[s0], boxes.VAL.iloc[int(ex.box_seq_start)] + 0.5 * atr[s0])

# ---- 6. single-bar departures ----
sb1 = f[f.acc_gap == 2]
print("\n[6] filled trades with exactly one fully-above-band bar between tests:", len(sb1))
ex = sb1.iloc[0]
for j in range(int(ex.acc_t1_last), int(ex.acc_t2_start) + 1):
    print("     ", T(j), "state", st[j], "H/L/C", h[j], l[j], c[j], "band", round(zlo[j], 2), round(zhi[j], 2), "atr", round(atr[j], 2))
print("      bounce (ATR) =", round(ex.acc_bounce_atr, 3), " signal", ex.signal_time)

# ---- 7. bounce reference: band at departure bar vs band at test-2 bar ----
br = f[f.acc_bounce_vs_d0_band_atr < 1.0]
print("\n[7] filled trades accepted only because the bounce is measured against the band top at the test-2 bar (vs departure bar):", len(br))
print(br[["signal_time", "acc_d0", "acc_t2_start", "acc_zh_d0", "acc_zh_t2", "acc_max_high", "acc_bounce_atr", "acc_bounce_vs_d0_band_atr", "acc_box_d0", "acc_box_t2", "acc_gap"]].round(3).to_string())

# ---- 8. band moved (box carry / ATR) -> touch or departure created by band movement ----
bm = f[(~f.acc_t2_low_vs_prev_bar_band) | (f.acc_d0_low_vs_prev_bar_band == False)]
print("\n[8] filled trades where the second-test touch or the departure exists only because the band moved:", len(bm))
print(bm[["signal_time", "acc_d0", "acc_t2_start", "acc_t2_low_vs_prev_bar_band", "acc_d0_low_vs_prev_bar_band", "acc_box_d0", "acc_box_t2"]].to_string())
for _, r in bm.iterrows():
    for j in (int(r.acc_d0) - 1, int(r.acc_d0), int(r.acc_t2_start) - 1, int(r.acc_t2_start)):
        print("     ", T(j), "box", sbox[j], "state", st[j], "L", l[j], "band_hi", round(zhi[j], 2), "atr", round(atr[j], 2))

# ---- 9. box change: second-test touch not in the sequence-start box's band ----
bc = f[~f.t2_touch_vs_seqstart_band]
print("\n[9] filled trades whose second-test touch is not a touch of the sequence-start box's band:", len(bc))
print(bc[["signal_time", "box_seq_start", "box_signal", "val_at_seq_start", "VAL", "atr", "t2_low", "vah_at_seq_start", "VAH"]].round(2).to_string())
f.to_csv(f"{Dd}/s03_filled_diag.csv", index=False)

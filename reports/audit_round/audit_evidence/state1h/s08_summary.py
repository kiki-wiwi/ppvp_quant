"""s08: recompute every figure cited in the state1h findings from the saved BASELINE artifacts
(baseline_out.pkl from s00, s01_signal_diag.csv / s01_events.json from s01, s05_running.npz from s05,
s07_requirement_fates.csv from s07). Baseline only; IS only; no variant is run or counted.
Run order: s00 -> s01 -> s05 -> s07 -> s08 (s03/s06 print the worked examples).
"""
import json
import pickle
import sys

sys.path.insert(0, "/home/user/ppvp_quant")
import numpy as np
import pandas as pd

from config import IS_END
from ppvp_audit.engine_path import tick_size

Dd = "/tmp/claude-0/-home-user-ppvp-quant/04a66f10-1e5e-5960-9a9f-ffb5b694e43a/scratchpad/audit_round/state1h"
P = pickle.load(open(f"{Dd}/baseline_out.pkl", "rb"))
df1h, sig, boxes, trades = P["df1h"], P["signals"], P["boxes"], P["trades"]
assert df1h.index.max() < IS_END and len(df1h) == 35064
dg = pd.read_csv(f"{Dd}/s01_signal_diag.csv")
ev = json.load(open(f"{Dd}/s01_events.json"))
R = np.load(f"{Dd}/s05_running.npz")
t1b, stb, st = R["t1_before"], R["st_before"], R["state"]
assert np.array_equal(st, sig.state.to_numpy())
led = pd.read_csv("/home/user/ppvp_quant/reports/audit_round/signal_opportunities_B_original.csv")
assert np.array_equal(led.signal_bar.to_numpy(), dg.signal_bar.to_numpy())
h, l, c, o = (df1h[k].to_numpy(float) for k in ("high", "low", "close", "open"))
zlo, zhi, atr, val = (sig[k].to_numpy() for k in ("zone_lo", "zone_hi", "atr", "VAL"))
brk, sraw, sbox = sig.breakdown.to_numpy(), sig.sig_raw.to_numpy(), sig.box_id.to_numpy()
bval, bvah = boxes.VAL.to_numpy(), boxes.VAH.to_numpy()
n = len(df1h)
F = dg.status == "filled"
f = dg[F].copy()
print("raw signals", len(dg), "| filled", int(F.sum()), "| status", led.status.value_counts().to_dict())
print("events (s01):", ev)

print("\n== S-1 separation ==")
print("filled: by bounce", int(f.acc_by_bounce.sum()), "| by bars", int(f.acc_by_bars.sum()),
      "| bars-only", int((~f.acc_by_bounce & f.acc_by_bars).sum()), "| bounce-only", int((f.acc_by_bounce & ~f.acc_by_bars).sum()))
print("all 120: bars-only", int((~dg.acc_by_bounce & dg.acc_by_bars).sum()), "status of bars-only:",
      dg.loc[~dg.acc_by_bounce & dg.acc_by_bars, "status"].tolist(), "bounce", dg.loc[~dg.acc_by_bounce & dg.acc_by_bars, "acc_bounce_atr"].round(3).tolist())
print("filled min bounce", round(f.acc_bounce_atr.min(), 4), "| <1.10 ATR", int((f.acc_bounce_atr < 1.10).sum()), "| <1.05 ATR", int((f.acc_bounce_atr < 1.05).sum()))
print("filled single fully-above bar between tests (gap==2):", int((f.acc_gap == 2).sum()), "| gap quantiles", f.acc_gap.quantile([0, .25, .5]).to_dict())
print("filled: bounce vs band top at departure bar < 1 ATR:", int((f.acc_bounce_vs_d0_band_atr < 1).sum()))
print("filled with >=1 merge:", int((f.n_merge > 0).sum()), "| merge dist", f.n_merge.value_counts().sort_index().to_dict())
merge = (stb == 3) & (st == 2)
print("IS merges", int(merge.sum()), "| merged touch set a lower first-test low", int((merge & (l < t1b)).sum()))
dep = (stb == 2) & (st == 3); roll = (stb == 4) & (st == 3); t2 = (stb == 3) & ((st == 4) | ((st == 1) & sraw))
print("TOUCH1->DEPARTED", int(dep.sum()), "all low>band top", bool((l[dep] > zhi[dep]).all()),
      "| rolls", int(roll.sum()), "all low>band top", bool((l[roll] > zhi[roll]).all()),
      "| DEPARTED->TOUCH2", int(t2.sum()), "all low<=band top", bool((l[t2] <= zhi[t2]).all()))
ok = np.array_equal(led.t1_last_bar, dg.acc_t1_last) and np.array_equal(led.departure_bar, dg.acc_d0) \
    and np.array_equal(led.touch_gap_bars_from_t1_last, dg.acc_gap) and np.allclose(led.departure_height_atr, dg.acc_bounce_atr, atol=1e-9)
mine = np.where(dg.acc_by_bounce & dg.acc_by_bars, "bounce+bars", np.where(dg.acc_by_bounce, "bounce", np.where(dg.acc_by_bars, "bars_only", "unexplained")))
print("ledger departure columns match independent machine:", ok, "| qualified_by labels match:", bool((mine == led.second_test_qualified_by).all()))

print("\n== S-2 carry ==")
s = f.acc_t2_start.to_numpy().astype(int); sb = f.signal_bar.to_numpy().astype(int)
chg = f.box_signal != f.box_seq_start
print("filled with box change during the sequence", int(chg.sum()), "| carry dist", f.n_carry.value_counts().sort_index().to_dict(),
      "| ledger profile_changed_between_tests (filled)", int(led.profile_changed_between_tests[F].sum()))
dval = ((f.VAL - f.val_at_seq_start) / f.atr)[chg].abs()
dvah = ((f.VAH / f.vah_at_seq_start - 1) * 100)[chg]
print("|dVAL|/ATR(signal bar) max", round(dval.max(), 3), "median", round(dval.median(), 3), "| dVAH% min/max", round(dvah.min(), 2), round(dvah.max(), 2))
old_zh_t2 = f.val_at_seq_start.to_numpy() + 0.5 * atr[s]
print("filled whose second-test start is not a touch of the sequence-start box band:", int((l[s] > old_zh_t2).sum()))
print("filled where departure or second touch exists only relative to a moved band:",
      int(((~f.acc_t2_low_vs_prev_bar_band) | (f.acc_d0_low_vs_prev_bar_band == False)).sum()),
      "| of which box switch between the two bars:",
      int((((~f.acc_t2_low_vs_prev_bar_band) & (sbox[s] != sbox[s - 1])) |
           ((f.acc_d0_low_vs_prev_bar_band == False) & (sbox[f.acc_d0.astype(int)] != sbox[f.acc_d0.astype(int) - 1]))).sum()))
print("filled whose first-test low lies above the signal-bar band top:", int((f.t1_low > f.zh).sum()),
      "| below the signal-bar band low:", int((f.t1_low < f.zl).sum()))

print("\n== S-3 roll ==")
print("filled roll dist", f.n_roll.value_counts().sort_index().to_dict())
rl = f[(f.n_roll > 0) & (f.orig_all_low < np.minimum(f.t1_low, f.t2_low))]
gap_atr = ((rl.orig_all_low - 0.2 * rl.atr) - rl.sl).abs() / rl.atr
gap_pct = ((rl.orig_all_low - 0.2 * rl.atr) - rl.sl).abs() / rl.close * 100
print("rolled with an earlier lower test low", len(rl), "| SL above all-tests stop by ATR", round(gap_atr.min(), 3), "-", round(gap_atr.max(), 3),
      "| % of close", round(gap_pct.min(), 3), "-", round(gap_pct.max(), 3))

print("\n== S-4 first test at activation inside band ==")
cat = np.where(f.seq_start_prev_low_above_zh, "prev bar fully above band",
               np.where(f.seq_start_prev_close_vs_zh_atr > 0, "prev close above band", "prev close inside band"))
print(pd.crosstab(cat, f.seq_start_at_activation))
ib = (cat == "prev close inside band")
print("inside-band activation starts among filled", int(ib.sum()), "| without roll", int((ib & (f.n_roll == 0)).sum()),
      "| IS test-1 starts", ev["t1_start"], "of which at activation with prior close inside band", ev["t1_start_at_activation_in_band"])

print("\n== S-5 pattern lag ==")
print("filled lag dist", f.lag_from_t2_start.value_counts().sort_index().to_dict(), "| lag>=1", int((f.lag_from_t2_start >= 1).sum()))

print("\n== S-6 reclaim ==")
print("filled close<VAL", int(f.close_below_VAL.sum()), "| close<first-test low", int(f.close_below_t1_low.sum()),
      "| either", int((f.close_below_VAL | f.close_below_t1_low).sum()), "| both", int((f.close_below_VAL & f.close_below_t1_low).sum()))

print("\n== S-7 touch geometry ==")
rows = []
for _, r in f.iterrows():
    bars = np.arange(int(r.seq_start_bar), int(r.acc_t1_last) + 1)
    b1 = bars[st[bars] == 2]
    rows.append(int(np.sum(h[b1] < zlo[b1])))
print("filled with a test-1 bar entirely below the band", int((np.array(rows) > 0).sum()),
      "| with a test-2 bar entirely below the band", int((f.t2_bars_below_band > 0).sum()))
depth = (f.zl - f.t1_low) / f.atr
print("first-test low below signal-bar band low (ATR): max", round(depth.max(), 2), "| stop distance (close-SL)/ATR max", round(((f.close - f.sl) / f.atr).max(), 2))

print("\n== S-8 consumption / requirements ==")
fa = pd.read_csv(f"{Dd}/s07_requirement_fates.csv")
print(fa.groupby(["origin", "fate"]).size().to_dict())
print("signal requirements claimed by >=1 later same-zone box:", int(((fa.origin == "signal") & (fa.claimed_box_switches > 0)).sum()),
      "| bars to fate (signal) median/max", fa[fa.origin == "signal"].eval("fate_bar-write_bar").median(), fa[fa.origin == "signal"].eval("fate_bar-write_bar").max())

print("\n== S-9 INVALID ==")
occ = pd.Series(st).value_counts(normalize=True).sort_index().round(4).to_dict()
print("IS occupancy by state code", occ)
prev = np.r_[0, st[:-1]]; bsw = np.r_[False, sbox[1:] != sbox[:-1]]
into5 = np.isin(prev, (1, 2, 3, 4)) & (st == 5)
print("transitions into INVALID from ARMED/TOUCH1/DEPARTED/TOUCH2", int(into5.sum()), "| close-based breakdown bar", int((into5 & brk).sum()),
      "| box switch (memory block)", int((into5 & ~brk & bsw).sum()), "| unexplained", int((into5 & ~brk & ~bsw).sum()))
canc = np.isin(prev, (2, 3, 4)) & (st == 5)
print("active sequences ended by INVALID", int(canc.sum()), "| breakdown bars in IS", int(brk.sum()), "| any closing >= band low", int((brk & (c >= zlo)).sum()))

print("\n== S-10 engulfing prior low ==")
print("signals with engulfing", int((dg.pattern & 2 > 0).sum()), "| prior low used in SL", int(dg.eng_prev_used.sum()))

print("\n== B1 code path (baseline only) ==")
tick = tick_size(df1h.index).astype(float)
m = np.isin(stb, (3, 4)) & ~np.isnan(t1b) & ((l <= t1b - tick) | (l < zlo)) & (c >= np.fmax(val, t1b))
print("DEPARTED/TOUCH2 bars piercing first-test low (>=1 tick) or band low, close >= max(VAL, first-test low):", int(m.sum()),
      "| -> INVALID", int((st[m] == 5).sum()), "| breakdown flag", int(brk[m].sum()),
      "| -> ARMED not explained by signal/box switch", int((m & (st == 1) & ~sraw & ~bsw).sum()))

print("\n== SL / TP ==")
tr = trades.set_index("entry_bar")
print("trades SL/TP == signal SL/TP at entry bar:", np.array_equal(tr.sl.to_numpy(), sig.sl.to_numpy()[tr.index]),
      np.array_equal(tr.tp.to_numpy(), sig.tp.to_numpy()[tr.index]))
i = df1h.index.get_loc(pd.Timestamp("2023-03-07 00:00", tz="UTC"))
print("hand check 2023-03-07 00:00 bar: t1_low", sig.t1_low.iloc[i], "ATR", round(atr[i], 4), "-> SL", round(sig.t1_low.iloc[i] - 0.2 * atr[i], 2),
      "kernel", round(sig.sl.iloc[i], 2), "| TP", round(sig.tp.iloc[i], 2), "= VAH box", int(sbox[i]), round(bvah[sbox[i]], 2), "| VAH previous box", round(bvah[sbox[i] - 1], 2))

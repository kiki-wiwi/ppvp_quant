"""s05: which box (profile version) was active at the first test (start and low bar), the second test start and the
signal, for all 120 raw signals / 109 IS trades; cross-checked against signal_opportunities_B_original.csv."""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/ppvp_quant")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from strategy_state_machine import state_machine_args, _run_state_machine
from config import BASELINE_STRATEGY as S
from sm_instr import run_sm_instr

d = pickle.load(open(HERE / "is_objects.pkl", "rb"))
df1h, boxes, sig, trades = d["df1h"], d["boxes"], d["sig"], d["trades"]
args = state_machine_args(df1h, S, boxes)
ref = _run_state_machine(*args)
ins = run_sm_instr(*args)
same = all(np.array_equal(a, b, equal_nan=(np.asarray(a).dtype.kind == "f")) for a, b in zip(ref, ins[:15]))
print("instrumented copy reproduces all 15 engine outputs bit for bit:", same)
(state, zlo, zhi, brk, sig_raw, pat, sl, tp, t1_low, t1_bar, t2_low, t2_bar, t2_start, sbox, need,
 seq_start, seq_box, seq_val, t2_val, carry, chain_start, nroll, cdist) = ins

bars = np.flatnonzero(sig_raw)
avail = pd.DatetimeIndex(boxes["available_time"])
val = boxes["VAL"].to_numpy()
idx = df1h.index
rows = []
for s in bars:
    ss = seq_start[s]
    seg = sbox[ss:s + 1]
    nsw = int((np.diff(seg) != 0).sum())
    ncar = int(carry[ss + 1:s + 1].sum())
    rows.append({"signal_bar": s, "signal_open": idx[s], "seq_start_bar": ss, "seq_start_time": idx[ss],
                 "t1_bar": t1_bar[s], "t2_start_bar": t2_start[s],
                 "box_at_seq_start": sbox[ss], "seq_box_recorded": seq_box[s], "box_at_t1_low": sbox[t1_bar[s]],
                 "box_at_t2_start": sbox[t2_start[s]], "box_at_signal": sbox[s],
                 "n_box_switches_in_sequence": nsw, "n_carry_switches": ncar,
                 "seq_val_at_signal": seq_val[s], "VAL_signal_box": val[sbox[s]],
                 "VAL_first_test_box": val[sbox[ss]], "atr_signal": args[5][s],
                 "hours_box_avail_to_signal_open": (idx[s] - avail[sbox[s]]) / pd.Timedelta(hours=1),
                 "chain_start_bar": chain_start[s], "n_rolls": int(nroll[s]), "box_at_chain_start": sbox[chain_start[s]],
                 "state_at_seq_start": int(state[ss])})
v = pd.DataFrame(rows)
assert (v["box_at_seq_start"] == v["seq_box_recorded"]).all()
opp = pd.read_csv("/home/user/ppvp_quant/reports/audit_round/signal_opportunities_B_original.csv")
assert (opp["signal_bar"].to_numpy() == v["signal_bar"].to_numpy()).all()
print("CSV profile_id == box at signal:", bool((opp["profile_id"].to_numpy() == v["box_at_signal"]).all()))
print("CSV profile_id_at_t1 == box at first-test LOW bar:", bool((opp["profile_id_at_t1"].to_numpy() == v["box_at_t1_low"]).all()))
print("CSV profile_id_at_t2_start == box at t2_start:", bool((opp["profile_id_at_t2_start"].to_numpy() == v["box_at_t2_start"]).all()))
v["status"] = opp["status"].to_numpy()
v["first_test_start_ne_low_box"] = v["box_at_seq_start"] != v["box_at_t1_low"]
v["csv_profile_changed_between_tests"] = opp["profile_changed_between_tests"].to_numpy()
v["changed_incl_first_test_start"] = (v["box_at_seq_start"] != v["box_at_signal"]) | (v["box_at_t1_low"] != v["box_at_signal"]) | (v["box_at_t2_start"] != v["box_at_signal"])
v["any_switch_in_sequence"] = v["n_box_switches_in_sequence"] > 0
tr = v[v["status"] == "filled"]
print("\n109 trades:")
print("  all four boxes identical (start, low, t2, signal):", int((~tr["any_switch_in_sequence"]).sum()))
print("  box changed somewhere between first-test start and signal:", int(tr["any_switch_in_sequence"].sum()))
print("  CSV flag profile_changed_between_tests True:", int(tr["csv_profile_changed_between_tests"].sum()))
print("  changed incl. first-test start (start/low/t2 vs signal):", int(tr["changed_incl_first_test_start"].sum()))
print("  box at first-test START differs from box at first-test LOW bar:", int(tr["first_test_start_ne_low_box"].sum()))
print("  box at t2_start differs from box at signal:", int((tr["box_at_t2_start"] != tr["box_at_signal"]).sum()))
print("  box at first-test start differs from signal box:", int((tr["box_at_seq_start"] != tr["box_at_signal"]).sum()))
print("  n box switches in sequence distribution:", tr["n_box_switches_in_sequence"].value_counts().sort_index().to_dict())
print("  all in-sequence switches were carries:", bool((tr["n_box_switches_in_sequence"] == tr["n_carry_switches"]).all()))
dv = (tr["VAL_signal_box"] - tr["VAL_first_test_box"]).abs() / tr["atr_signal"]
print("  |VAL(signal box) - VAL(first-test-start box)| / ATR_signal for changed: ",
      dv[tr["box_at_seq_start"] != tr["box_at_signal"]].describe().round(3).to_dict())
print("  hours from signal-box available_time to signal bar open: min %.0f median %.0f max %.0f" %
      (tr["hours_box_avail_to_signal_open"].min(), tr["hours_box_avail_to_signal_open"].median(), tr["hours_box_avail_to_signal_open"].max()))
print("  rows where seq_start != t1_bar (low not on the first bar of test 1):", int((tr["seq_start_bar"] != tr["t1_bar"]).sum()))
print(tr[tr["first_test_start_ne_low_box"] | (tr["box_at_t2_start"] != tr["box_at_signal"])][
    ["signal_open", "seq_start_time", "box_at_seq_start", "box_at_t1_low", "box_at_t2_start", "box_at_signal",
     "n_box_switches_in_sequence", "csv_profile_changed_between_tests"]].to_string())
print("  trades whose signalling first test is a rolled failed second test:", int((tr["n_rolls"] > 0).sum()),
      " rolls distribution:", tr["n_rolls"].value_counts().sort_index().to_dict())
print("  of those, box at chain start (original ARMED->TOUCH1 bar) differs from signal box:",
      int(((tr["n_rolls"] > 0) & (tr["box_at_chain_start"] != tr["box_at_signal"])).sum()))
print("  chain spans >1 profile version (chain start, first-test start, low, t2 start vs signal):",
      int(((tr["box_at_chain_start"] != tr["box_at_signal"]) | tr["changed_incl_first_test_start"]).sum()))
print("  chain length hours (chain start -> signal): median %.0f max %.0f" % ((tr.signal_bar - tr.chain_start_bar).median(), (tr.signal_bar - tr.chain_start_bar).max()))
# carry tolerance check at every box switch inside the 109 trade chains (switch while TOUCH1/DEPARTED/TOUCH2)
dists, kept = [], []
for _, t in tr.iterrows():
    for j in range(int(t.chain_start_bar) + 1, int(t.signal_bar) + 1):
        if not np.isnan(cdist[j]):
            dists.append(cdist[j]); kept.append(bool(carry[j]))
dists, kept = np.array(dists), np.array(kept)
print(f"  in-chain box switches: {len(dists)}; all carried: {kept.all()}; |VAL_new - seq_val|/ATR_switch max {dists.max():.3f} (rule: <= k1+k2 = 1.0)")
v.to_csv(HERE / "profile_versions_per_signal.csv", index=False)
pickle.dump({"ins": ins}, open(HERE / "sm_instr_out.pkl", "wb"))

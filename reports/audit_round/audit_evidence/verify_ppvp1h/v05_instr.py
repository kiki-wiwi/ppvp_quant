"""v05: my own instrumented copy of _run_state_machine (text edits on the live source) -> first-test start, chain start,
rolls, carry events; checks P-6 (ledger t1 semantics / flag) and P-8 (profile versions mixed by carry/roll). IS only."""
import inspect, pickle, sys, textwrap
from pathlib import Path
import numpy as np, pandas as pd
V = Path(__file__).resolve().parent; sys.path.insert(0, str(V)); sys.path.insert(0, "/home/user/ppvp_quant")
import strategy_state_machine as SM
from config import BASELINE_STRATEGY as S
D = pickle.load(open(V / "is_objs.pkl", "rb"))
df1h, boxes, sig, tr = D["df1h"], D["boxes"], D["sig"], D["trades"]
src = inspect.getsource(SM._run_state_machine.py_func)
src = src.replace("@njit(cache=True)\n", "")
def sub(old, new):
    global src
    assert src.count(old) == 1, old
    src = src.replace(old, new)
sub("    t2_val = np.nan           #", "    t1_start = -1; chain_start = -1; n_roll = 0\n"
    "    o_t1s = np.full(n, -1, np.int64); o_chs = np.full(n, -1, np.int64); o_nr = np.zeros(n, np.int64)\n"
    "    o_carry = np.zeros(n, np.bool_); o_cdv = np.full(n, np.nan)\n    t2_val = np.nan           #")
sub("                        keep = True", "                        keep = True\n                        o_carry[i] = True; o_cdv[i] = abs(box_val[b] - seq_val) / a")
sub("                if hv_block or not keep:\n", "                if hv_block or not keep:\n                    t1_start = -1; chain_start = -1; n_roll = 0\n")
sub("            elif touched:\n                st = 2\n", "            elif touched:\n                st = 2\n                t1_start = i; chain_start = i; n_roll = 0\n")
sub("                if roll:\n                    st = 3\n", "                if roll:\n                    st = 3\n                    t1_start = t2_start; n_roll += 1\n")
sub("                sig[i] = True\n", "                sig[i] = True\n                o_t1s[i] = t1_start; o_chs[i] = chain_start; o_nr[i] = n_roll\n"
    "                t1_start = -1; chain_start = -1; n_roll = 0\n")
sub("            s_box, s_need)", "            s_box, s_need, o_t1s, o_chs, o_nr, o_carry, o_cdv)")
ns_ = {"np": np}
exec(textwrap.dedent(src), ns_)
args = SM.state_machine_args(df1h, S, boxes)
res = ns_["_run_state_machine"](*args)
ref = SM._run_state_machine(*args)
names = ("state zlo zhi brk sig pat sl tp t1_low t1_bar t2_low t2_bar t2_start box need").split()
same = all(np.array_equal(x, y, equal_nan=True) for x, y in zip(res[:15], ref))
print("instrumented copy reproduces all 15 kernel outputs bit for bit:", same)
assert same
state, sbox_all = res[0], res[13]
t1s, chs, nr, carry, cdv = res[15:]
assert np.array_equal(sbox_all, sig["box_id"].to_numpy())
eb = tr["entry_bar"].to_numpy(int); allsig = np.flatnonzero(res[4])
t1b, t2s = res[9], res[12]
atr = sig["atr"].to_numpy(); VAL = boxes["VAL"].to_numpy(); VAH = boxes["VAH"].to_numpy(); av = boxes["available_time"]
def rows(bars):
    out = []
    for s in bars:
        a, ch, lo, t2 = t1s[s], chs[s], t1b[s], t2s[s]
        bx = sbox_all
        sw_t1 = int((bx[a+1:s+1] != bx[a:s]).sum()); sw_ch = int((bx[ch+1:s+1] != bx[ch:s]).sum())
        cin = np.flatnonzero(carry[ch+1:s+1]) + ch + 1
        out.append(dict(signal_bar=s, signal_open=df1h.index[s], t1_start=df1h.index[a], chain_start=df1h.index[ch], n_roll=nr[s],
                        box_chain=bx[ch], box_t1start=bx[a], box_t1low=bx[lo], box_t2start=bx[t2], box_sig=bx[s],
                        sw_from_t1start=sw_t1, sw_from_chain=sw_ch, n_carry_in_chain=len(cin),
                        max_carry_dv=float(cdv[cin].max()) if len(cin) else np.nan,
                        flag_csv=bool(bx[lo] != bx[s] or bx[t2] != bx[s]),
                        dVAL_t1=abs(VAL[bx[s]] - VAL[bx[a]]) / atr[s], dVAH_t1=abs(VAH[bx[s]] - VAH[bx[a]]) / atr[s],
                        first_bar_of_box=bool(df1h.index[s] == av[bx[s]])))
    return pd.DataFrame(out)
T = rows(eb)
print("\n109 trades:")
print("  any box switch between first-test START and signal:", int((T.sw_from_t1start > 0).sum()), " distribution", T.sw_from_t1start.value_counts().sort_index().to_dict())
print("  CSV-style flag (box at t1 LOW bar or t2_start != signal box):", int(T.flag_csv.sum()))
miss = T[(T.sw_from_t1start > 0) & ~T.flag_csv]
print("  switched but flag False:", len(miss)); print(miss[["signal_open", "t1_start", "box_t1start", "box_t1low", "box_t2start", "box_sig"]].to_string(index=False))
print("  flag True but no switch from t1 start:", int((T.flag_csv & (T.sw_from_t1start == 0)).sum()))
print("  box at t1 start != box at t1 low:", int((T.box_t1start != T.box_t1low).sum()))
print("  rolled (n_roll>0):", int((T.n_roll > 0).sum()), T.n_roll.value_counts().sort_index().to_dict())
print("  rolled with box at chain start != signal box:", int(((T.n_roll > 0) & (T.box_chain != T.box_sig)).sum()))
print("  chain spans >1 profile version (any switch chain start..signal):", int((T.sw_from_chain > 0).sum()))
print("  in-chain carry switches:", int(T.n_carry_in_chain.sum()), "| max |VAL_new-seq_val|/ATR at switch: %.3f" % np.nanmax(T.max_carry_dv))
# all in-chain switches are carries? count switches in chains vs carry events
tot_sw = int(T.sw_from_chain.sum()); print("  total in-chain box switches:", tot_sw, "== carry events:", tot_sw == int(T.n_carry_in_chain.sum()))
ch = T[T.sw_from_t1start > 0]
print("  changed (n=%d): |dVAL| ATR median %.2f max %.2f ; |dVAH| ATR median %.2f P90 %.2f max %.2f" % (len(ch), ch.dVAL_t1.median(), ch.dVAL_t1.max(),
      ch.dVAH_t1.median(), ch.dVAH_t1.quantile(.9), ch.dVAH_t1.max()))
print("  signalled on the first bar that could use the signal box:", int(T.first_bar_of_box.sum()))
for t in ("2020-01-22 19:00", "2020-05-31 23:00", "2021-06-04 08:00"):
    r = T[T.signal_open == pd.Timestamp(t, tz="UTC")].iloc[0]
    print(f"  {t}: chain_start {r.chain_start} t1_start {r.t1_start} n_roll {r.n_roll} boxes chain/t1start/t1low/t2/sig {r.box_chain}/{r.box_t1start}/{r.box_t1low}/{r.box_t2start}/{r.box_sig} flag {r.flag_csv}")
k0, k1, k2 = 400, 401, 402; s = T[T.signal_open == pd.Timestamp("2021-06-04 08:00", tz="UTC")].signal_bar.iloc[0]
print(f"  box 400 VAL {VAL[k0]:.1f} VAH {VAH[k0]:.1f}; 401 VAL {VAL[k1]:.1f} VAH {VAH[k1]:.1f}; 402 VAL {VAL[k2]:.1f} VAH {VAH[k2]:.1f} avail {av[k2]}; ATR {atr[s]:.1f};"
      f" TP-VAH401 {(VAH[k2]-VAH[k1])/atr[s]:+.2f} TP-VAH400 {(VAH[k2]-VAH[k0])/atr[s]:+.2f}")
# CSV check
op = pd.read_csv("/home/user/ppvp_quant/reports/audit_round/signal_opportunities_B_original.csv")
A = rows(allsig)
print("\n120 signals: CSV profile_id/at_t1/at_t2_start equal to as-of mapping:",
      bool((op.profile_id.to_numpy() == A.box_sig.to_numpy()).all() and (op.profile_id_at_t1.to_numpy() == A.box_t1low.to_numpy()).all()
           and (op.profile_id_at_t2_start.to_numpy() == A.box_t2start.to_numpy()).all()),
      "| CSV t1_bar == kernel t1 LOW bar:", bool((op.t1_bar.to_numpy() == t1b[allsig]).all()),
      "| CSV t1_bar == first-test start:", int((op.t1_bar.to_numpy() == t1s[allsig]).sum()), "of 120")
fl = op.loc[op.status == "filled", "profile_changed_between_tests"].sum(); print("CSV filled profile_changed_between_tests:", int(fl))
# boxes used by the trades (chain start .. signal, every bar), and how many were INVALID while current
used = set()
for _, r in T.iterrows():
    used |= set(sbox_all[df1h.index.get_loc(r.chain_start):r.signal_bar + 1].tolist())
used4 = set(T.box_chain) | set(T.box_t1start) | set(T.box_t1low) | set(T.box_t2start) | set(T.box_sig)
inv = set(np.unique(sbox_all[(state == 5) & (sbox_all >= 0)]).tolist())
print("distinct boxes at (chain start, t1 start, t1 low, t2 start, signal):", len(used4), "| any bar chain..signal:", len(used),
      "| of the 4-point set INVALID while current at some time:", len(used4 & inv))
T.to_csv(V / "v05_trades_profile_versions.csv", index=False)

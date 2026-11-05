"""s09: which trades make up the figure-1 ambiguity counts (IS baseline)."""
import sys
sys.path.insert(0, "/home/user/ppvp_quant")
import numpy as np, pandas as pd
from ppvp_audit.repro_figures import baseline_frames
out, tf = baseline_frames()
h = out["df1h"]["high"].to_numpy(float)
xb = tf["exit_bar"].to_numpy(int)
sl = (tf["exit_reason"] == "SL").to_numpy()
amb_tp = sl & (h[xb] >= tf["tp"].to_numpy())
in_left = sl & tf["poc_above_entry"].to_numpy() & tf["mfe_r_strict"].notna().to_numpy()
poc_exit = sl & tf["poc_above_entry"].to_numpy() & (h[xb] >= tf["poc"].to_numpy())
not_strict = in_left & ~tf["reached_poc_strict"].to_numpy(dtype=bool)
two = np.flatnonzero(not_strict & poc_exit); one = np.flatnonzero(amb_tp)
print("SL&TP trade index:", one.tolist(), " SL&POC (left panel, not strict) trade indices:", two.tolist())
print("distinct trades behind the 1 + 2 counts:", sorted(set(one.tolist()) | set(two.tolist())))
for j in sorted(set(one.tolist()) | set(two.tolist()) | set(np.flatnonzero(poc_exit).tolist())):
    print(f"  trade {j}: entry {tf['entry_time'].iat[j]} hold {xb[j]-tf['entry_bar'].iat[j]} amb_tp {bool(amb_tp[j])} poc_in_exit_bar {bool(poc_exit[j])} "
          f"in_left_panel {bool(in_left[j])} reached_poc_strict {bool(tf['reached_poc_strict'].iat[j])}")

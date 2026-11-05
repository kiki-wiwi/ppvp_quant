"""s07: matched event-level null (ppvp_audit/nulls.py) on the baseline, IS only: reproduce the published numbers and check
the candidate-set rules independently (|c-b|>24, <=30/60/90 days, same SMA200 side & vol tercile, baseline flat at c's close,
at least one later bar). No variant is run."""
import sys, json
sys.path.insert(0, "/home/user/ppvp_quant")
import numpy as np, pandas as pd
from ppvp_audit import nulls as N
from ppvp_audit.repro_figures import baseline_frames
out, tf = baseline_frames()
df = out["df1h"]; n = len(df)
ev = N.matched_event_null(out, tf)
pub = json.load(open("/home/user/ppvp_quant/reports/audit_round/random_baseline_review.json"))["matched_event"]
print("reproduced:", {k: round(ev[k], 6) for k in ("null_mean_r_mean", "actual_mean_r", "p_lower")},
      " published:", {k: round(pub[k], 6) for k in ("null_mean_r_mean", "actual_mean_r", "p_lower")})
feats = N.bar_features(out)
sets, used = N.candidate_sets(tf["entry_bar"].to_numpy(), feats, df)
t = df.index.as_unit("ns").asi8
tr = out["result"].trades
inpos = np.zeros(n, bool)
for e, x in zip(tr["entry_bar"], tr["exit_bar"]):
    inpos[int(e):int(x)] = True
bad = 0
for b, pool, w in zip(tf["entry_bar"].to_numpy(), sets, used):
    ok = ((np.abs(pool - b) > 24) & (np.abs(t[pool] - t[b]) <= w * 86400 * 10**9) & (pool < n - 1) & ~inpos[pool]
          & (feats["above_sma200"].to_numpy()[pool] == feats["above_sma200"].to_numpy()[b])
          & (feats["vol_tercile"].to_numpy()[pool] == feats["vol_tercile"].to_numpy()[b]))
    bad += int((~ok).sum())
print("candidate bars violating a stated rule:", bad, "; pool sizes min/median/max", min(map(len, sets)), int(np.median(list(map(len, sets)))), max(map(len, sets)),
      "; windows used", pd.Series(used).value_counts().to_dict())
# is the real entry bar ever 'flat at close'? (the baseline enters at that close, so it is in position at the close)
print("real entry bars marked flat_at_close:", int(feats['flat_at_close'].to_numpy()[tf['entry_bar'].to_numpy()].sum()))

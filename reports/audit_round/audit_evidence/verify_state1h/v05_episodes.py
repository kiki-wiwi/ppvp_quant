"""v05: S-1 near-threshold sensitivity and plain-sense pullback episodes (runs of touched bars) per filled trade. BASELINE, IS only."""
import pickle, numpy as np, pandas as pd
from collections import Counter
D = "/tmp/claude-0/-home-user-ppvp-quant/04a66f10-1e5e-5960-9a9f-ffb5b694e43a/scratchpad/audit_round/verify_state1h/"
O = pickle.load(open(D + "v00_objs.pkl", "rb")); T = pickle.load(open(D + "v01_trace.pkl", "rb"))
l = O["df1h"]["low"].to_numpy(float); zhi = T["zhi"]
R = pd.read_csv(D + "v02_signals.csv"); F = R[R.filled]
near = F[(F.bounce < 1.10)]
print("S-1 filled bounce<1.10:", len(near), "| of which bounce-only (gap<6):", int((~near.by_bars).sum()))
print("S-1 filled bounce-only (gap<6):", int((F.by_bounce & ~F.by_bars).sum()))
ep = []
for r in F.itertuples():
    bars = np.arange(r.cs, r.s + 1); t = l[bars] <= zhi[bars]
    ep.append(int(t[0]) + int(((t[1:]) & (~t[:-1])).sum()))
F = F.assign(episodes=ep)
print("plain-sense pullback episodes per filled trade (runs of touched bars from chain start to signal):", dict(sorted(Counter(ep).items())))
print("trades on exactly the 2nd episode:", int((F.episodes == 2).sum()), "| >=3 episodes:", int((F.episodes >= 3).sum()),
      "| >=3 with no roll (merges only):", int(((F.episodes >= 3) & (F.n_roll == 0)).sum()))

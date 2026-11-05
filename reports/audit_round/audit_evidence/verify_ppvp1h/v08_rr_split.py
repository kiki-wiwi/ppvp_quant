"""v08: P-6 downstream effect: share of 'tests spanned profile versions' in the RR<1 / RR>=1 groups, CSV flag vs first-test-start definition."""
import pickle
from pathlib import Path
import numpy as np, pandas as pd
V = Path(__file__).resolve().parent
D = pickle.load(open(V / "is_objs.pkl", "rb")); tr = D["trades"]
T = pd.read_csv(V / "v05_trades_profile_versions.csv")
assert (T.signal_bar.to_numpy() == tr.entry_bar.to_numpy()).all()
rr = ((tr.tp - tr.entry_px) / (tr.entry_px - tr.sl)).to_numpy()
for name, m in (("RR<1", rr < 1), ("RR>=1", rr >= 1)):
    print(f"{name}: n {int(m.sum())}  CSV flag share {T.flag_csv[m].mean():.3f} ({int(T.flag_csv[m].sum())})  "
          f"switch-from-first-test-start share {(T.sw_from_t1start[m] > 0).mean():.3f} ({int((T.sw_from_t1start[m] > 0).sum())})  "
          f"switch-from-chain-start share {(T.sw_from_chain[m] > 0).mean():.3f} ({int((T.sw_from_chain[m] > 0).sum())})")

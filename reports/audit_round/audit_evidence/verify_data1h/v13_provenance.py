# D-3: which provenance fields exist at each layer (IS only)
import _setup
import pandas as pd
from ppvp_audit.common import is_run
out = is_run()
df, d4, bx, sig, tr = out["df1h"], out["df4h"], out["boxes"], out["signals"], out["result"].trades
print("1H columns:", list(df.columns))
print("1H fill_source values:", df["fill_source"].value_counts().to_dict())
print("4H columns:", list(d4.columns), "| n_bars counts:", d4["n_bars"].value_counts().to_dict())
hrs = ((bx["window_end"] - bx["window_start"]) / pd.Timedelta(hours=1)).astype(int)
print("boxes:", len(bx), "| n_1h == window hours for all:", bool((bx["n_1h"] == hrs).all()), "| box columns:", list(bx.columns))
prov = [c for c in sig.columns if any(k in c for k in ("fill", "source", "n_bars", "partial", "imputed", "missing", "venue"))]
print("signal-table provenance-like columns:", prov)
for f in ("reports/audit_round/signal_opportunities_B_original.csv", "reports/audit_round/trades_B_original.csv"):
    cols = pd.read_csv(f, nrows=1).columns
    print(f.rsplit("/", 1)[-1], "provenance-like columns:", [c for c in cols if any(k in c for k in ("fill_source", "source", "n_bars", "partial", "imputed", "missing", "venue", "spot"))])

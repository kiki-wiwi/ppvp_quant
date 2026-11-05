# (1) The two signal rows whose SL changes when the partial hour 2021-03-02 01:00 is treated as missing: magnitude, trades?
# (2) What a NaN-volume (filled) 1H bar does to volume_profile (code path of the empty scale-ratio window).
# (3) The sanctioned IS entry points make no network call (requests is blocked).
import numpy as np
import pandas as pd
import requests
from ppvp_audit.common import is_run
from config import BASELINE_STRATEGY as S
from ppvp_engine import build_boxes, volume_profile
from strategy_state_machine import generate_signals

out = is_run()
df1h, sig, tr = out["df1h"], out["signals"], out["result"].trades
t = pd.Timestamp("2021-03-02 01:00", tz="UTC")
d2 = df1h.drop(index=t)
bx2, d42 = build_boxes(d2, S)
sg2 = generate_signals(d2, S, bx2, d42, None)
a, b = sig.loc[sg2.index, "sl"], sg2["sl"]
diff = ~((a == b) | (a.isna() & b.isna()))
entry_times = set(tr["entry_time"] - pd.Timedelta(hours=1))
for ts in a.index[diff]:
    print("SL differs at signal bar", ts, "baseline %.10f dropped %.10f rel %.3e" % (a[ts], b[ts], b[ts] / a[ts] - 1),
          "| is a traded signal:", ts in entry_times, "| sig_raw", bool(sig.loc[ts, "sig_raw"]), "signal", bool(sig.loc[ts, "signal"]))

# (2) NaN volume in a profile window
lo = np.array([100.0, 101.0, 102.0, 103.0]); hi = lo + 2.0
vol = np.array([10.0, 50.0, 10.0, 10.0])
good = volume_profile(lo, hi, vol, 50, 0.7)
vol_nan = vol.copy(); vol_nan[3] = np.nan
bad = volume_profile(lo, hi, vol_nan, 50, 0.7)
print("\nvolume_profile finite volumes: POC %.3f VAL %.3f VAH %.3f" % (good["poc"], good["val"], good["vah"]))
print("volume_profile one NaN volume : POC %.3f VAL %.3f VAH %.3f  (NaN bins: %d of %d)" % (
      bad["poc"], bad["val"], bad["vah"], int(np.isnan(bad["hist"]).sum()), len(bad["hist"])))
v = pd.Series([1.0] * 30); v.iloc[10] = np.nan
print("20-bar volume SMA after one NaN volume: NaN for", int(v.rolling(20, min_periods=20).mean().iloc[10:].isna().sum()), "bars")

# (3) block all HTTP and call the sanctioned entry points from a cold pipeline cache
import pipeline
pipeline.clear_cache()
def blocked(*a, **k):
    raise RuntimeError("network blocked by audit script")
requests.Session.request = blocked
requests.get = blocked
from ppvp_audit.common import load_is_market
d, f, s = load_is_market()
o2 = is_run()
print("\nload_is_market + is_run with HTTP blocked: OK (1H rows %d, funding rows %d, spot days %d, trades %d)" % (
      len(d), len(f), len(s), len(o2["result"].trades)))

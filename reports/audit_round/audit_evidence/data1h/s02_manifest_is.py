# Print only the IS-relevant (< 2024-01-01) entries of data/manifest.json QA for BTCUSDT.
# Aggregate statistics computed over the full 2020-2026 window are deliberately NOT printed.
import json
import pandas as pd
IS_END = pd.Timestamp("2024-01-01", tz="UTC")
m = json.load(open("/home/user/ppvp_quant/data/manifest.json"))
print("top-level keys:", list(m.keys()))
print("timestamp_convention:", m.get("timestamp_convention"))
print("cleaning_policy:", json.dumps(m.get("cleaning_policy"), ensure_ascii=False, indent=1))
qa = m["qa"]["BTCUSDT"]
print("BTC qa keys:", list(qa.keys()))
def before(ts):
    return pd.Timestamp(ts) < IS_END
print("dropped placeholder bars < 2024:", [t for t in qa["dropped_placeholder_bars"] if before(t)])
print("archive gap fills starting < 2024:", [g for g in qa["archive_gap_fills"] if before(g["gap_start"])])
for k in ("perp_1h_raw_archive", "perp_1h_cleaned", "perp_4h_official", "spot_1h"):
    d = qa[k]
    print(k, "keys:", list(d.keys()))
    print("   missing ranges starting < 2024:", [r for r in d["missing_ranges"] if before(r[0])])
adj = qa["resampled_4h_vs_official_4h_adjudication_1m"]
print("adjudication records < 2024:")
for r in adj:
    if before(r["bar_open_time"]):
        print("  ", json.dumps(r))
print("resampled_4h_vs_official_4h keys (values not printed: full-window aggregates):", list(qa["resampled_4h_vs_official_4h"].keys()))
ds = m["datasets"]
print("dataset keys:", list(ds.keys()))
f = ds["BTCUSDT_funding"]["files"]
print("funding files listed:", len(f), "first:", f[0]["url"].split("/")[-1], f[0].get("rows"))

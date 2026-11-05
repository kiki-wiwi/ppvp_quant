# IS-only view of data/manifest.json: prints only records dated < 2024-01-01; never prints 1m-derived values
import json
import pandas as pd
IS_END = pd.Timestamp("2024-01-01", tz="UTC")
m = json.load(open("data/manifest.json"))
print("cleaning_policy:", json.dumps(m["cleaning_policy"], ensure_ascii=False, indent=1))
ds = m["datasets"]["BTCUSDT_perp_1h"]
ph = [t for t in ds["dropped_placeholder_bars"] if pd.Timestamp(t) < IS_END]
gf = [g for g in ds["gap_fills"] if pd.Timestamp(g["gap_start"]) < IS_END]
print("IS dropped placeholders:", ph)
print("IS gap fills:", gf)
qa = m["qa"]["BTCUSDT"]
print("qa keys:", list(qa.keys()))
adj = qa["resampled_4h_vs_official_4h_adjudication_1m"]
for r in adj:
    t = pd.Timestamp(r["bar_open_time"])
    if t < IS_END:
        print("IS adjudication record:", r["bar_open_time"], "keys:", sorted(r.keys()), "verdict:", r["verdict"])
        print("   resampled_1h:", r["resampled_1h"]); print("   official_4h:", r["official_4h"])
        print("   has official_1m_aggregate key:", "official_1m_aggregate" in r, "(values not printed)")

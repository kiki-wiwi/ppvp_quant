# D-12: does reports/audit_round/baseline_manifest.json carry metadata of rows at/after IS_END?  Prints booleans/keys only.
import _setup
import json
import pandas as pd
from _setup import IS_END
m = json.load(open("reports/audit_round/baseline_manifest.json"))
print("segment:", m["segment"], "| definition mentions IS:", "IS [2020-01-01, 2024-01-01)" in m["definition"])
for name, meta in m["data"]["datasets_used"].items():
    keys = sorted(meta.keys())
    last_post = pd.Timestamp(meta["last"]) >= IS_END
    more_rows = meta["rows_total"] > meta["rows_used_in_this_round"]
    gf = meta.get("gap_fills_in_file") or []
    ph = meta.get("dropped_placeholder_bars_in_file") or []
    gf_post = any(pd.Timestamp(g["gap_start"]) >= IS_END for g in gf)
    ph_post = any(pd.Timestamp(t) >= IS_END for t in ph)
    gf_ratio = any("volume_scale_ratio_binance_over_okx" in g for g in gf)
    print(f"{name}: keys={keys}\n   'last' at/after IS_END: {last_post} | rows_total > rows_used: {more_rows} | "
          f"post-IS gap-fill record present: {gf_post} (carries volume ratio: {gf_ratio}) | post-IS placeholder present: {ph_post}")

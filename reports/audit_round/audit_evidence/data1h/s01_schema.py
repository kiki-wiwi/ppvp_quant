# Schema-only inspection of the BTC parquet caches (no row values printed beyond the IS window).
import pyarrow.parquet as pq
import pandas as pd
ROOT = "/home/user/ppvp_quant/data/processed/"
IS_END = pd.Timestamp("2024-01-01", tz="UTC")
for name in ["BTCUSDT_perp_1h", "BTCUSDT_funding", "BTCUSDT_spot_1h", "BTCUSDT_perp_4h_official"]:
    f = pq.ParquetFile(ROOT + name + ".parquet")
    print(name, "schema:")
    print("  ", f.schema_arrow)
    md = f.schema_arrow.metadata or {}
    print("   pandas metadata keys:", list(md.keys()))
    df = pd.read_parquet(ROOT + name + ".parquet")
    df = df[df.index < IS_END]          # truncate immediately
    print("   IS rows:", len(df), "index name:", df.index.name, "tz:", df.index.tz, "unit:", df.index.dtype)
    print("   IS first/last index:", df.index.min(), df.index.max())
    print("   dtypes:", dict(df.dtypes.astype(str)))
    if "fill_source" in df.columns:
        print("   fill_source counts (IS):", df["fill_source"].value_counts().to_dict())
    print()

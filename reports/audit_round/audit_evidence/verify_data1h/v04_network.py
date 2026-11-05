# D-4: (1) sanctioned entry points run with all sockets blocked; (2) a missing parquet routes to build_cache();
# (3) fetch_vision_zip requests the .CHECKSUM over the network even when the zip is cached. No network is touched.
import _setup
import tempfile
from pathlib import Path
import pandas as pd
import data_loader as DL
import pipeline
from ppvp_audit.common import is_run, load_is_market

df1h, fund, spot = load_is_market()
out = is_run()
tr = out["result"].trades
print("(1) with sockets blocked: 1H rows", len(df1h), "funding rows", len(fund), "spot warm-up days", len(spot),
      "| is_run trades", len(tr), "reasons", tr["exit_reason"].value_counts().to_dict(),
      "| IS Sharpe %.3f" % out["report"]["IS"]["sharpe"], "| network attempts", len(_setup.NET_ATTEMPTS))

# (2) missing cache file -> build_cache() (stubbed so nothing is downloaded or written)
called = []
def build_stub(*a, **k):
    called.append("build_cache")
    raise RuntimeError("build_cache would run here (downloads all archives, OKX REST, 1m daily archives)")
DL.build_cache = build_stub
tmp = Path(tempfile.mkdtemp(dir="/tmp/claude-0/-home-user-ppvp-quant/04a66f10-1e5e-5960-9a9f-ffb5b694e43a/scratchpad/audit_round/verify_data1h"))
orig = DL.PROC_DIR
DL.PROC_DIR = tmp                       # pretend data/processed is empty
pipeline.clear_cache()
try:
    load_is_market()
except RuntimeError as e:
    print("(2) load_is_market with an empty processed dir ->", called, "|", str(e)[:80])
DL.PROC_DIR = orig
pipeline.clear_cache()

# (3) fetch_vision_zip asks the network for the .CHECKSUM before looking at the local zip
seen = []
def http_stub(url, params=None, **k):
    seen.append(url)
    raise RuntimeError("stubbed network")
DL._http_get = http_stub
rel = "futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2020-01.zip"
print("(3) local zip exists:", (DL.RAW_DIR / rel).exists())
try:
    DL.fetch_vision_zip(rel)
except RuntimeError:
    pass
print("    first URL requested by fetch_vision_zip:", seen[0].rsplit('/', 1)[-1])
print("network attempts total", len(_setup.NET_ATTEMPTS))

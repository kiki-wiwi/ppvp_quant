# D-5 reachability: when do NaN-volume fill rows reach a volume profile? (synthetic, OKX stubbed, network blocked)
import _setup
import numpy as np
import pandas as pd
import data_loader as DL
from ppvp_audit.common import load_is_market
from ppvp_engine import build_boxes
from strategy_state_machine import generate_signals
from config import BASELINE_STRATEGY as S

df1h, _, _ = load_is_market()
H = pd.Timedelta(hours=1)
RAWCOLS = ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]
x = df1h.loc["2020-01-01":"2020-01-31", RAWCOLS].copy()
def okx_stub(symbol, start, end, bar="1H"):
    idx = pd.date_range(start, end - H, freq="h", tz="UTC")
    src = df1h.reindex(idx)
    return pd.DataFrame({"open": src["open"].to_numpy(), "high": src["high"].to_numpy(), "low": src["low"].to_numpy(),
                         "close": src["close"].to_numpy(), "volume": 2.0, "quote_volume": 2.0 * src["close"].to_numpy()},
                        index=pd.DatetimeIndex(idx, name="open_time"))
def okx_stub_nohist(symbol, start, end, bar="1H"):      # OKX returns nothing for the pre-gap window
    o = okx_stub(symbol, start, end, bar)
    return o if (end - start) <= pd.Timedelta(hours=72) else o.iloc[0:0]
DL.RESEARCH_END = pd.Timestamp("2020-02-01", tz="UTC")
for label, stub, gap_start, n in (("start-of-window", okx_stub, "2020-01-01 00:00", 6),
                                  ("start-of-window", okx_stub, "2020-01-01 00:00", 24),
                                  ("mid-month, OKX history empty", okx_stub_nohist, "2020-01-15 00:00", 6)):
    DL.fetch_okx_klines = stub
    gap = pd.date_range(gap_start, periods=n, freq="h", tz="UTC")
    f, rec = DL.fill_archive_gaps(x.drop(index=gap), "BTCUSDT")
    f = f.drop(columns=["fill_source"])
    bx, d4 = build_boxes(f, S)
    nanbox = bx[~np.isfinite(bx["total_volume"])]
    sig = generate_signals(f, S, bx, d4, None)
    print(f"{label} gap {gap_start} x{n}h: ratio {rec[0]['volume_scale_ratio_binance_over_okx']}, NaN-volume rows {int(f['volume'].isna().sum())}, "
          f"boxes {len(bx)}, boxes with NaN volume {len(nanbox)}, of which VAL==P_low {int((nanbox['VAL'] == nanbox['P_low']).sum())}, "
          f"vsma NaN bars after warm-up {int(sig['vsma'].iloc[21:].isna().sum())}")
print("network attempts", len(_setup.NET_ATTEMPTS))

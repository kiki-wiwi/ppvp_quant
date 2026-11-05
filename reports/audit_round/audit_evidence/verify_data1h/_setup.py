"""Common setup for verify_data1h scripts: repo on sys.path, all network access blocked and logged."""
import os
import socket
import sys

REPO = "/home/user/ppvp_quant"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

NET_ATTEMPTS = []


class NetworkBlocked(RuntimeError):
    pass


def _blocked(*a, **k):
    NET_ATTEMPTS.append(repr(a)[:200])
    raise NetworkBlocked(f"network access blocked in verify_data1h: {a!r}"[:300])


socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked

import pandas as pd  # noqa: E402

IS_START = pd.Timestamp("2020-01-01", tz="UTC")
IS_END = pd.Timestamp("2024-01-01", tz="UTC")


def read_spot_is():
    """Binance spot 1H from the local parquet, truncated to < IS_END immediately (no later row is touched)."""
    s = pd.read_parquet(os.path.join(REPO, "data/processed/BTCUSDT_spot_1h.parquet"))
    s = s[s.index < IS_END].copy()
    assert s.index.max() < IS_END
    return s

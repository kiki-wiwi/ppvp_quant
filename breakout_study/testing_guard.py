"""Explicit pytest plugin: prohibit market-file reads and network in synthetic runs.

Load with ``pytest -p breakout_study.testing_guard ...``. This is not a global
conftest: it cannot change the behavior of other studies' normal test suites.
"""
import os
from pathlib import Path
import sys


def _market_path(path):
    if not isinstance(path,(str,bytes,os.PathLike)):
        return False
    s=os.fsdecode(path).replace('\\','/')
    return any(part in '/'+s for part in (
        '/data/raw/', '/data/processed/', '/data/processed_1m/'))


def pytest_sessionstart(session):
    def audit(event,args):
        if event=='open' and args and _market_path(args[0]):
            raise PermissionError('synthetic test guard: market file access prohibited')
        if event=='socket.connect':
            raise PermissionError('synthetic test guard: network access prohibited')
    sys.addaudithook(audit)
    import pandas as pd
    read=pd.read_parquet
    def guarded(path,*args,**kwargs):
        if _market_path(path):
            raise PermissionError('synthetic test guard: market parquet access prohibited')
        return read(path,*args,**kwargs)
    pd.read_parquet=guarded

"""Explicit CP1 plugin: block market files and network before test collection."""
import os
import sys

def _market_path(path):
    if not isinstance(path,(str,bytes,os.PathLike)): return False
    text=os.fsdecode(path).replace('\\','/')
    return any(part in '/'+text for part in ('/data/raw/','/data/processed/','/data/processed_1m/',
        '/data/venue_market/'))

def pytest_sessionstart(session):
    def audit(event,args):
        if event=='open' and args and _market_path(args[0]):
            raise PermissionError('CP1 guard: market file access prohibited')
        if event=='socket.connect': raise PermissionError('CP1 guard: network access prohibited')
    sys.addaudithook(audit)
    import pandas as pd
    original=pd.read_parquet
    def guarded(path,*args,**kwargs):
        if _market_path(path): raise PermissionError('CP1 guard: market parquet access prohibited')
        return original(path,*args,**kwargs)
    pd.read_parquet=guarded

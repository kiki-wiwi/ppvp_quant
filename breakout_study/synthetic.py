"""Deterministic invented prices only; no reference to market files."""
import numpy as np
import pandas as pd
from .config import IS_START, IS_END
from .scope import make_context


def study_context():
    """Full registered calendar with an invented monthly breakout and reversal."""
    index=pd.date_range(IS_START,IS_END,freq='min',inclusive='left').as_unit('ns')
    n=len(index)
    close=np.full(n,100.)
    volume=np.full(n,10.)
    for month in pd.date_range(IS_START,IS_END,freq='MS',inclusive='left'):
        start=index.get_loc(month+pd.Timedelta(days=9,hours=12))
        close[start:start+60]=np.linspace(100.,104.,60)
        volume[start:start+60]=30.
        close[start+60:start+780]=np.linspace(104.,110.,720)
        close[start+780:start+960]=np.linspace(110.,100.,180)
    opened=np.r_[close[0],close[:-1]]
    frame=pd.DataFrame({'open':opened,'high':np.maximum(opened,close)+.1,
        'low':np.minimum(opened,close)-.1,'close':close,'volume':volume,
        'bad':False,'archive_gap':False},index=index)
    frame.attrs['provenance']='synthetic'
    funding=pd.Series(.0001,index=pd.date_range(IS_START,IS_END,freq='8h',inclusive='left'))
    return make_context(frame,funding)

"""Invented full-calendar fixture; no market files, fitted values, or RNG."""
import numpy as np
import pandas as pd
from . import config,scope


def study_context():
    index=pd.date_range(config.IS_START,config.IS_END,freq='min',inclusive='left').as_unit('ns')
    close=np.full(len(index),10000.)
    # One artificial rise each month. Mixed follow-through creates nondegenerate
    # synthetic ledgers without using observed price or return information.
    for j,month in enumerate(pd.date_range(config.IS_START,config.IS_END,freq='MS',inclusive='left')):
        start=index.get_loc(month+pd.Timedelta(days=6,hours=12))
        close[start:start+60]=np.linspace(10000.,10400.,60)
        close[start+60:start+122]=np.linspace(10400.,10400.+(j%3-1)*80.,62)
    opened=np.r_[close[0],close[:-1]]
    frame=pd.DataFrame(dict(open=opened,high=np.maximum(opened,close)+10.,
        low=np.minimum(opened,close)-10.,close=close,volume=100.,
        bad=(index.hour==14)&(index.minute==0),archive_gap=False),index=index)
    # Artificial one-minute daily maintenance bounds random holding time and
    # exercises forced exits. These are explicitly not registered real gaps.
    frame.loc[frame.bad,'volume']=0.
    frame.loc[pd.Timestamp('2021-05-07 13:30',tz='UTC'),'archive_gap']=True
    frame.attrs['provenance']='synthetic'
    funding=pd.Series(.00001,index=pd.date_range(config.IS_START,config.IS_END,freq='8h',inclusive='left'))
    return scope.make_context(frame,funding)

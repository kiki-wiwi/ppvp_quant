"""Independent stage locks and immutable venue-specific minute snapshots."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import ASSETS, VENUES, utc

BTC_IS_UNLOCKED = True
POST_IS_AND_ETH_UNLOCKED = True
MINUTE_NS = 60_000_000_000
class ScopeError(ValueError): pass
class RealDataLockedError(PermissionError): pass

def require_real_authorization(asset,venue,start,end):
    if asset not in ASSETS or venue not in VENUES: raise ScopeError('unregistered market')
    start,end=utc(start),utc(end)
    if start>=end: raise ScopeError('empty/reversed request')
    post=asset=='ETH' or end>pd.Timestamp('2024-01-01',tz='UTC')
    if post:
        if not POST_IS_AND_ETH_UNLOCKED: raise RealDataLockedError('POST-IS / ETH remains independently locked')
    elif not BTC_IS_UNLOCKED: raise RealDataLockedError('BTC IS remains locked')
    earliest={('BTC','UM'):'2020-01-01',('BTC','SPOT'):'2020-01-01',('BTC','CM'):'2020-08-11',
              ('ETH','UM'):'2020-07-01',('ETH','SPOT'):'2020-07-01',('ETH','CM'):'2020-08-18'}[(asset,venue)]
    if start<pd.Timestamp(earliest,tz='UTC') or end>pd.Timestamp('2026-11-01',tz='UTC'):
        raise ScopeError('request exceeds approved native-history bounds')

def utc_index(values,label='index'):
    idx=pd.DatetimeIndex(values)
    if idx.tz is None or idx.hasnans: raise ScopeError(f'{label}: aware finite times required')
    idx=idx.tz_convert('UTC').as_unit('ns')
    if idx.has_duplicates or not idx.is_monotonic_increasing: raise ScopeError(f'{label}: unique ordered times required')
    return idx

def _readonly(values,dtype):
    return np.frombuffer(np.asarray(values,dtype=dtype).tobytes(),dtype=dtype)

def assert_scope(ctx):
    if ctx.asset not in ASSETS or ctx.venue not in VENUES: raise ScopeError('unregistered market')
    if ctx.source=='real':
        require_real_authorization(ctx.asset,ctx.venue,ctx.index[0],ctx.index[-1]+pd.Timedelta(minutes=1))
    elif ctx.source!='synthetic': raise ScopeError('explicit synthetic or real provenance required')

@dataclass(frozen=True)
class Context:
    index: pd.DatetimeIndex
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray
    bad: np.ndarray
    archive_gap: np.ndarray
    funding_rate: np.ndarray
    source: str='synthetic'
    asset: str='BTC'
    venue: str='UM'
    def __len__(self): return len(self.index)
    def __post_init__(self):
        idx=utc_index(self.index)
        if not len(idx) or np.any(idx.asi8%MINUTE_NS) or np.any(np.diff(idx.asi8)!=MINUTE_NS):
            raise ScopeError('nonempty continuous minute grid required')
        assert_scope(self)
        values=_readonly(idx.asi8,'<i8').view('datetime64[ns]')
        dates=pd.arrays.DatetimeArray._simple_new(values,dtype=pd.DatetimeTZDtype('ns','UTC'))
        object.__setattr__(self,'index',pd.DatetimeIndex(dates,copy=False))
        for key in ('O','H','L','C','V','bad','archive_gap','funding_rate'):
            val=np.asarray(getattr(self,key))
            if val.shape!=(len(idx),): raise ScopeError(f'{key}: wrong shape')
            dtype=np.bool_ if key in ('bad','archive_gap') else np.float64
            if dtype==np.bool_ and not np.isin(val,[False,True]).all(): raise ScopeError('boolean flag required')
            object.__setattr__(self,key,_readonly(val,dtype))
        for key in ('O','H','L','C','V'):
            if not np.isfinite(getattr(self,key)).all(): raise ScopeError('finite price/volume required')
        if np.any(self.V<0) or np.any(np.minimum.reduce([self.O,self.H,self.L,self.C])<=0): raise ScopeError('invalid price/volume')
        valid=~self.bad
        if np.any(self.H[valid]<np.maximum(self.O[valid],self.C[valid])) or np.any(self.L[valid]>np.minimum(self.O[valid],self.C[valid])):
            raise ScopeError('invalid OHLC bounds')
        if np.isinf(self.funding_rate).any(): raise ScopeError('infinite funding rate')
        if self.venue=='SPOT' and np.isfinite(self.funding_rate).any(): raise ScopeError('spot cannot contain funding')

def make_context(frame,funding=None,*,source='synthetic',asset='BTC',venue='UM'):
    if source=='real':
        # Locks are checked before inspecting prices; real loaders must also call this before file I/O.
        if asset=='ETH':
            if not POST_IS_AND_ETH_UNLOCKED: raise RealDataLockedError('ETH locked')
        elif not BTC_IS_UNLOCKED and not POST_IS_AND_ETH_UNLOCKED: raise RealDataLockedError('all real input locked')
    if frame.attrs.get('source')=='real' and source!='real': raise ScopeError('cannot relabel real provenance')
    if source=='synthetic' and frame.attrs.get('provenance')!='synthetic': raise ScopeError('explicit synthetic provenance required')
    idx=utc_index(frame.index)
    if source=='real': require_real_authorization(asset,venue,idx[0],idx[-1]+pd.Timedelta(minutes=1))
    rates=np.full(len(idx),np.nan)
    if funding is not None and len(funding):
        original=utc_index(funding.index,'funding input')
        rounded=utc_index(original.round('min'),'funding rounded')
        where=idx.get_indexer(rounded)
        vals=np.asarray(funding,float)
        if (where<0).any() or vals.shape!=(len(rounded),) or not np.isfinite(vals).all(): raise ScopeError('invalid funding grid')
        rates[where]=vals
    cols=('open','high','low','close','volume')
    if not set(cols).issubset(frame): raise ScopeError('OHLCV required')
    bad=frame['bad'].to_numpy() if 'bad' in frame else np.zeros(len(idx),bool)
    gap=frame['archive_gap'].to_numpy() if 'archive_gap' in frame else np.zeros(len(idx),bool)
    return Context(idx,*(frame[c].to_numpy() for c in cols),bad,gap,rates,source,asset,venue)

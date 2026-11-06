"""IS-only immutable inputs and a lock independent of the old CP3 engine."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import IS_START, IS_END

REAL_DATA_UNLOCKED = False
MINUTE_NS = 60_000_000_000


class ScopeError(ValueError):
    pass


class RealDataLockedError(PermissionError):
    pass


def require_real_authorization():
    if not REAL_DATA_UNLOCKED:
        raise RealDataLockedError('Real market input is locked: this stage permits synthetic validation only.')


def utc_index(values, label='index'):
    idx = pd.DatetimeIndex(values)
    if idx.tz is None or idx.hasnans:
        raise ScopeError(f'{label}: finite timezone-aware timestamps required')
    idx = idx.tz_convert('UTC').as_unit('ns')
    if idx.has_duplicates or not idx.is_monotonic_increasing:
        raise ScopeError(f'{label}: unique ascending timestamps required')
    if len(idx) and (idx[0] < IS_START or idx[-1] >= IS_END):
        raise ScopeError(f'{label}: only BTC IS [2020,2024) is allowed')
    return idx


def _readonly(values, dtype):
    # Immutable bytes backing prevents even setflags(write=True) on this snapshot.
    arr = np.asarray(values, dtype=dtype)
    return np.frombuffer(arr.tobytes(), dtype=dtype)


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
    source: str = 'synthetic'
    symbol: str = 'BTCUSDT'

    def __len__(self):
        return len(self.index)

    def __post_init__(self):
        require_synthetic(self)
        idx = utc_index(self.index)
        if not len(idx) or np.any(idx.asi8 % MINUTE_NS) or np.any(np.diff(idx.asi8) != MINUTE_NS):
            raise ScopeError('context requires a nonempty continuous minute grid')
        object.__setattr__(self, 'index', pd.DatetimeIndex(_readonly(idx.asi8, '<i8'), tz='UTC'))
        for key in ('O','H','L','C','V','bad','archive_gap','funding_rate'):
            val = np.asarray(getattr(self, key))
            if val.shape != (len(idx),):
                raise ScopeError(f'{key}: length/shape mismatch')
            dtype = np.bool_ if key in ('bad','archive_gap') else np.float64
            if dtype == np.bool_ and not np.isin(val, [False, True]).all():
                raise ScopeError(f'{key}: boolean values required')
            object.__setattr__(self, key, _readonly(val, dtype))
        for key in ('O','H','L','C','V'):
            if not np.isfinite(getattr(self, key)).all():
                raise ScopeError(f'{key}: all grid prices/volume must be finite')
        if np.any(self.V < 0) or np.any(np.minimum.reduce([self.O,self.H,self.L,self.C]) <= 0):
            raise ScopeError('positive prices and nonnegative volume required')
        valid = ~self.bad
        if np.any(self.H[valid] < np.maximum(self.O[valid],self.C[valid])) or np.any(self.L[valid] > np.minimum(self.O[valid],self.C[valid])):
            raise ScopeError('invalid OHLC bounds')
        if np.isinf(self.funding_rate).any():
            raise ScopeError('infinite funding rate')


def require_synthetic(ctx):
    if ctx.symbol != 'BTCUSDT':
        raise ScopeError('only BTCUSDT is allowed')
    if ctx.source == 'real':
        require_real_authorization()
    elif ctx.source != 'synthetic':
        raise ScopeError('explicit source=synthetic or real required')


def make_context(frame, funding=None, *, source='synthetic', symbol='BTCUSDT'):
    if source == 'real':
        require_real_authorization()  # Before inspecting caller-supplied market frames.
    if frame.attrs.get('source') == 'real' and source != 'real':
        raise ScopeError('real provenance cannot be relabelled synthetic')
    if source == 'synthetic' and frame.attrs.get('provenance') != 'synthetic':
        raise ScopeError('synthetic context requires an explicitly generated synthetic frame')
    idx = utc_index(frame.index)
    rates = np.full(len(idx), np.nan)
    if funding is not None and len(funding):
        original = pd.DatetimeIndex(funding.index)
        if original.tz is None:
            raise ScopeError('funding timestamps require timezone')
        rounded = utc_index(original.tz_convert('UTC').round('min'), 'funding')
        where = idx.get_indexer(rounded)
        if np.any(where < 0):
            raise ScopeError('funding outside supplied minute grid')
        values = np.asarray(funding, float)
        if values.shape != (len(rounded),) or not np.isfinite(values).all():
            raise ScopeError('finite funding values required')
        rates[where] = values
    cols = ('open','high','low','close','volume')
    if not set(cols).issubset(frame.columns):
        raise ScopeError('OHLCV columns required')
    arrays = [frame[c].to_numpy() for c in cols]
    bad = frame['bad'].to_numpy() if 'bad' in frame else np.zeros(len(idx), bool)
    gaps = frame['archive_gap'].to_numpy() if 'archive_gap' in frame else np.zeros(len(idx), bool)
    return Context(idx, *arrays, bad, gaps, rates, source, symbol)

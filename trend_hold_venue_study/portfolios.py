"""Independent Spot and inverse-contract accounts for the approved venue study.

Only caller-supplied immutable contexts are consumed. There is no file/network I/O.
Pressure events are an explicit simplified model, not exchange liquidation records.
"""
import math
import numpy as np
import pandas as pd

from .config import COSTS, INITIAL_CAPITAL, CONTRACT_SIZE, CM_VERSIONS
from .scope import Context, ScopeError, assert_scope
from .features import references
from .models import Result
from .accounting import initial_coin, spot_fill, trade_change, funding_change, pressure_event

_MINUTE = pd.Timedelta(minutes=1)
_LAST_INSTANT = _MINUTE - pd.Timedelta(nanoseconds=1)
_LD = np.longdouble


def _schedule(ctx, features, window, venue):
    if not isinstance(ctx, Context):
        raise ScopeError('validated immutable Context required')
    assert_scope(ctx)
    if ctx.venue != venue:
        raise ScopeError('wrong execution venue')
    index = ctx.index
    start, stop = index.searchsorted(window.start), index.searchsorted(window.end)
    if start >= len(index) or index[start] != window.start or stop <= start or index[stop-1] != window.end-_MINUTE:
        raise ScopeError('complete account-window minute grid required')
    days = pd.date_range(window.start, window.end, inclusive='left', freq='D').as_unit('ns')
    if len(days)*1440 != stop-start or not days.isin(features.index).all():
        raise ScopeError('complete account-window daily schedule required')
    if features.index.has_duplicates or features.columns.has_duplicates:
        raise ScopeError('duplicate canonical schedule labels')
    selected = features.loc[days].copy()
    if not {'g','budget','budget_update'}.issubset(selected.columns):
        raise ScopeError('canonical g/budget/budget_update required')
    for name in ('g','budget_update'):
        if not pd.api.types.is_bool_dtype(selected[name]) or selected[name].isna().any():
            raise ScopeError('complete boolean canonical schedule required')
    budget = selected['budget'].to_numpy(float)
    if not np.isfinite(budget).all() or np.any((budget <= 0)|(budget > 1)):
        raise ScopeError('invalid canonical risk budget')
    updates = (days.dayofweek == 0) | (days == window.signal_origin)
    if not np.array_equal(selected['budget_update'], updates):
        raise ScopeError('canonical budget update calendar mismatch')
    if np.any((budget[1:] != budget[:-1]) & ~updates[1:]):
        raise ScopeError('canonical budget changed outside registered update')
    if np.any(ctx.bad[start:stop].reshape(-1,1440).all(axis=1)):
        raise ScopeError('account contains a day without effective price')
    if ctx.bad[stop-1]:
        raise ScopeError('terminal minute must have an effective close')
    refs = references(ctx, days)
    if not np.isfinite(refs['eref']).all() or (refs['eref'] <= 0).any():
        raise ScopeError('invalid own-venue reference close')
    return start, stop, days, selected, refs


def _cost(cost):
    if COSTS.get(cost.name) != cost:
        raise ScopeError('unregistered cost')


def _daily(equity, days):
    values = np.asarray(equity, float)[1439::1440]
    previous = np.r_[INITIAL_CAPITAL,values[:-1]]
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ScopeError('nonfinite or negative daily equity')
    if np.any((previous == 0)&(values != 0)):
        raise ScopeError('absorbing zero-equity account resurrected')
    returns = np.zeros(len(values), float)
    nonzero = previous != 0
    returns[nonzero] = values[nonzero]/previous[nonzero]-1.
    return pd.Series(values,index=days,name='equity'),pd.Series(returns,index=days,name='return')


def run_spot(ctx, features, version, cost, window):
    """Unlevered Spot account; actual cash clips only the execution quantity."""
    _cost(cost)
    if version not in ('V2_SPOT','P_SPOT'):
        raise ScopeError('unregistered Spot version')
    start,stop,days,schedule,refs = _schedule(ctx,features,window,'SPOT')
    index = ctx.index[start:stop]
    rows = schedule.to_dict('records')
    eref = refs['eref'].to_numpy(float)
    qty,balance,previous_equity,previous_mark = 0.,INITIAL_CAPITAL,INITIAL_CAPITAL,eref[0]
    previous_g,pending = False,False
    orders,decisions,attribution = [],[],[]
    data = np.empty((len(index),6),float)
    held_qty = np.empty(len(index),float)
    total_terms = []
    for j,time in enumerate(index):
        i = start+j
        before_qty = qty
        open_ref = previous_mark if ctx.bad[i] else ctx.O[i]
        mark = previous_mark if ctx.bad[i] else ctx.C[i]
        price_open = _LD(qty)*(_LD(open_ref)-_LD(previous_mark))
        event_equity = _LD(balance)+_LD(qty)*_LD(open_ref)
        fee,slippage,roundoff = 0.,0.,0.
        if j % 1440 == 0:
            row = rows[j//1440]
            g = version == 'P_SPOT' or bool(row['g'])
            initial = j == 0
            flip = not initial and g != previous_g
            triggered = initial or bool(row['budget_update']) or flip or pending
            target = float(g*row['budget']*previous_equity/eref[j//1440]) if triggered else None
            decision = {**row,'time':time,'g':g,'initial':initial,'trend_flip':flip,'retry':pending,
                        'triggered':triggered,'equity_before':previous_equity,'qty_before':qty,
                        'qty_target':target,'reference_price':eref[j//1440]}
            status = 'hold'
            if triggered:
                pending = False
                if target == qty:
                    status = 'zero_delta'
                elif ctx.bad[i]:
                    status,pending = 'canceled_bad',True
                else:
                    fill = spot_fill(qty,balance,target,open_ref,cost)
                    orders.append({**fill,'time':time,'terminal':False,'qty_before':qty,
                                   'qty_target':target,'equity_before':previous_equity,'reference_price':open_ref})
                    qty,balance = fill['qty_after'],fill['cash_after']
                    fee,slippage,roundoff = fill['fee'],fill['slippage_cost'],fill['roundoff_adjustment']
                    status = 'executed'
            decision.update(status=status,pending_after=pending,qty_after=qty)
            decisions.append(decision)
            previous_g = g
        price_close = _LD(qty)*(_LD(mark)-_LD(open_ref))
        held_qty[j] = qty
        if j == len(index)-1 and qty:
            fill = spot_fill(qty,balance,0.,mark,cost)
            orders.append({**fill,'time':time+_LAST_INSTANT,'terminal':True,'qty_before':qty,
                           'qty_target':0.,'equity_before':float(_LD(balance)+_LD(qty)*_LD(mark)),
                           'reference_price':mark})
            qty,balance = fill['qty_after'],fill['cash_after']
            fee += fill['fee']; slippage += fill['slippage_cost']; roundoff += fill['roundoff_adjustment']
        marked = balance+qty*mark
        if not math.isfinite(marked) or marked <= 0 or balance < 0 or qty < 0:
            error = ScopeError(f'invalid Spot account at {time}')
            error.partial = {'orders':orders,'decisions':decisions,'attribution':attribution}
            raise error
        price_pnl = price_open+price_close
        total_terms.extend((price_pnl,-_LD(fee),-_LD(slippage),_LD(roundoff)))
        expected = _LD(previous_equity)+price_pnl-_LD(fee)-_LD(slippage)+_LD(roundoff)
        if abs(_LD(marked)-expected) >= _LD('1e-9'):
            raise ScopeError(f'Spot event reconciliation failed at {time}')
        attribution.append({'time':time,'price_pnl':price_pnl,'fee':fee,'slippage_cost':slippage,
                            'cash_roundoff_adjustment':roundoff})
        data[j] = (marked,qty,mark,balance,qty*mark/marked,before_qty)
        previous_equity,previous_mark = marked,mark
    error = float(_LD(data[-1,0])-_LD(INITIAL_CAPITAL)-np.sum(np.asarray(total_terms,dtype=np.longdouble),dtype=np.longdouble))
    if abs(error) >= 1e-9:
        raise ScopeError('Spot terminal attribution failed')
    minute = pd.DataFrame(data,index=index,columns=['equity','qty','mark_price','cash','actual_leverage','qty_before'])
    minute['held_qty'] = held_qty
    minute['bad'] = ctx.bad[start:stop]
    minute['archive_gap'] = ctx.archive_gap[start:stop]
    de,dr = _daily(minute['equity'],days)
    metadata = {'version':version,'cost':cost.name,'asset':ctx.asset,'window':window.name,
                'initial_capital':INITIAL_CAPITAL,'final_equity':float(data[-1,0]),
                'account_reconciliation_error':error,'order_count':len(orders),
                'adjustment_count':sum(not r['terminal'] for r in orders),
                'cash_clip_count':sum(bool(r['cash_clip']) for r in orders),
                'total_fees':math.fsum(r['fee'] for r in orders),
                'total_slippage_cost':math.fsum(r['slippage_cost'] for r in orders),
                'total_funding':0.,'liquidated':False}
    return Result(minute,de,dr,pd.DataFrame(orders),pd.DataFrame(decisions),pd.DataFrame(),metadata,
                  pd.DataFrame(attribution),pd.DataFrame())


_ATTR_COLUMNS = (
    'coin_before','coin_after','cm_before','cm_price','spot_before','spot_price',
    'contract_mtm_coin','funding_coin','slippage_coin','fee_coin','zero_floor_coin',
    'holding_spot_usd','holding_cm_usd','contract_mtm_spot_usd','funding_spot_usd',
    'slippage_spot_usd','fee_spot_usd','zero_floor_spot_usd','contract_mtm_cm_usd',
    'funding_cm_usd','slippage_cm_usd','fee_cm_usd','zero_floor_cm_usd')
_ATTR_KINDS = ('open','close','funding','trade','terminal_contract','pressure','pressure_spot_open')


class _CoinBook:
    """Realized wallet and harmonic inverse basis with independent equity recurrence."""
    def __init__(self, coin, price, spot, capacity):
        self.coin = self.wallet = _LD(coin)
        self.face = 0.
        self.basis = _LD(0.)  # Reciprocal harmonic entry price.
        self.price,self.spot = _LD(price),_LD(spot)
        self._times = np.empty(capacity,dtype=np.int64)
        self._kinds = np.empty(capacity,dtype=np.int8)
        self._values = np.empty((capacity,len(_ATTR_COLUMNS)),dtype=np.longdouble)
        self._count = 0
        self.max_wallet_error = 0.

    @property
    def rows(self):
        result = pd.DataFrame(self._values[:self._count],columns=_ATTR_COLUMNS,copy=False)
        result.insert(0,'kind',np.asarray(_ATTR_KINDS,dtype=object)[self._kinds[:self._count]])
        result.insert(0,'time',pd.to_datetime(self._times[:self._count],utc=True))
        return result

    def upnl(self):
        return _LD(self.face)*(self.basis-1/self.price) if self.face else _LD(0.)

    def assert_wallet(self):
        error = abs(self.coin-(self.wallet+self.upnl()))
        self.max_wallet_error = max(self.max_wallet_error,float(error))
        if not np.isfinite(self.coin) or not np.isfinite(self.wallet) or error > _LD('1e-12'):
            raise ScopeError('inverse wallet plus unrealized PnL reconciliation failed')

    def event(self, time, kind, price=None, spot=None, mtm=0., fund=0., slip=0., fee=0., floor=0.):
        price = self.price if price is None else _LD(price)
        spot = self.spot if spot is None else _LD(spot)
        previous_coin,previous_price,previous_spot = self.coin,self.price,self.spot
        change = _LD(mtm)+_LD(fund)+_LD(slip)-_LD(fee)+_LD(floor)
        self.coin += change
        self.price,self.spot = price,spot
        if self._count >= len(self._times):
            raise ScopeError('internal attribution event capacity exceeded')
        self._times[self._count] = time.value
        self._kinds[self._count] = _ATTR_KINDS.index(kind)
        self._values[self._count] = (
            previous_coin,self.coin,previous_price,price,previous_spot,spot,
            _LD(mtm),_LD(fund),_LD(slip),_LD(fee),_LD(floor),
            previous_coin*(spot-previous_spot),previous_coin*(price-previous_price),
            spot*_LD(mtm),spot*_LD(fund),spot*_LD(slip),spot*_LD(fee),spot*_LD(floor),
            price*_LD(mtm),price*_LD(fund),price*_LD(slip),price*_LD(fee),price*_LD(floor))
        self._count += 1

    def mark(self,time,price,spot,kind='mark'):
        price = _LD(price)
        change = _LD(self.face)*(1/self.price-1/price)
        self.event(time,kind,price,spot,mtm=change)
        self.assert_wallet()

    def fund(self,time,rate,cost):
        flow = funding_change(self.face,float(self.price),rate,cost)
        self.wallet += _LD(flow)
        self.event(time,'funding',fund=flow)
        self.assert_wallet()
        return flow

    def trade(self,time,target,cost,terminal=False):
        old = self.face
        delta = target-old
        calc = trade_change(delta,float(self.price),cost)
        fill = _LD(calc['fill_price'])
        # Track realized wallet against an independent mark-to-market recurrence.
        if not old or old*delta > 0:
            if target:
                self.basis = (_LD(old)*self.basis+_LD(delta)/fill)/_LD(target)
        else:
            closed = math.copysign(min(abs(delta),abs(old)),old)
            self.wallet += _LD(closed)*(self.basis-1/fill)
            if old*target < 0:
                self.basis = 1/fill
            elif target == 0:
                self.basis = _LD(0.)
        self.wallet -= _LD(calc['fee_coin'])
        self.face = float(target)
        # Apply the exact float64 accounting formula once; extended precision
        # is used only to accumulate balances and attribution, not to reprice.
        slip = calc['slippage_coin']
        self.event(time,'terminal_contract' if terminal else 'trade',slip=slip,fee=calc['fee_coin'])
        self.assert_wallet()
        return {**calc,'time':time,'terminal':terminal,'face_before':old,'face_target':target,
                'delta_face':delta,'reference_price':float(self.price),'slippage_coin':float(slip)}

    def liquidate(self,time,event):
        trigger = _LD(event['trigger_price'])
        mtm = _LD(self.face)*(1/self.price-1/trigger)
        raw = self.coin+mtm
        floor = max(_LD(0.),-raw)
        self.event(time,'pressure',price=trigger,mtm=mtm,floor=floor)
        self.face = 0.
        self.basis = _LD(0.)
        self.wallet = self.coin
        self.assert_wallet()
        return {**event,'time':time,'raw_coin':float(raw),'residual_coin':float(self.coin),
                'zero_floor_coin':float(floor),'residual_cm_usd':float(self.coin*trigger),
                'residual_spot_usd':float(self.coin*self.spot)}


def run_coin(cm, spot, features, version, cost, window, whole_contracts=False):
    """Inverse contract plus collateral coin; residual coins survive model pressure.

    At an opening collision, the frozen target remains recorded, old-position
    funding settles, then old-position opening pressure takes precedence over a
    voluntary rebalance. Surviving new positions face opening/extreme checks too.
    """
    _cost(cost)
    if version not in CM_VERSIONS:
        raise ScopeError('unregistered coin version')
    start,stop,days,schedule,cm_refs = _schedule(cm,features,window,'CM')
    ss,se,s_days,_,spot_refs = _schedule(spot,features,window,'SPOT')
    if cm.asset != spot.asset or cm.source != spot.source or not cm.index[start:stop].equals(spot.index[ss:se]) or not days.equals(s_days):
        raise ScopeError('coin collateral and contract contexts must share asset/provenance/window')
    if spot.bad[ss]:
        raise ScopeError('initial Spot conversion minute is bad')
    index = cm.index[start:stop]
    initial = initial_coin(INITIAL_CAPITAL,spot.O[ss],cost)
    estimate = initial_coin(INITIAL_CAPITAL,float(spot_refs['eref'].iloc[0]),cost)
    capacity = len(index)+2*np.count_nonzero(np.isfinite(cm.funding_rate[start:stop]))+3*len(days)+20
    book = _CoinBook(initial,float(cm_refs['eref'].iloc[0]),spot.O[ss],capacity)
    initial_spot = book.coin*_LD(spot.O[ss])-_LD(INITIAL_CAPITAL)
    initial_cm = book.coin*book.price-_LD(INITIAL_CAPITAL)
    previous_g,pending,liquidated = False,False,False
    rows = schedule.to_dict('records')
    orders,decisions,funding,pressure = [],[],[],[]
    data = np.empty((len(index),12),float)
    previous_spot = float(spot_refs['eref'].iloc[0])
    terminal_fx = _LD(0.)
    for j,time in enumerate(index):
        i,k = start+j,ss+j
        before_coin = float(book.coin)
        before_face = book.face
        terminal = j == len(index)-1
        target,decision = None,None
        if j % 1440 == 0:
            row = rows[j//1440]
            g = version == 'P_CM_DeltaMatched' or bool(row['g'])
            first = j == 0
            flip = not first and g != previous_g
            triggered = (first or bool(row['budget_update']) or flip or pending) and version != 'HODL_1x_Coin' and not liquidated
            decision_coin = estimate if first else before_coin
            own_ref = float(cm_refs['eref'].iloc[j//1440])
            if triggered:
                multiplier = g*row['budget'] if version == 'V4_CM_CoinAccum' else g*row['budget']-1.
                target = float(multiplier*decision_coin*own_ref)
                if whole_contracts:
                    target = math.trunc(target/CONTRACT_SIZE[cm.asset])*CONTRACT_SIZE[cm.asset]
            decision = {**row,'time':time,'g':g,'initial':first,'trend_flip':flip,'retry':pending,
                        'triggered':triggered,'coin_before':decision_coin,'actual_coin_before':before_coin,
                        'face_before':before_face,'face_target':target,'reference_price':own_ref,
                        'initial_coin_estimate':estimate if first else None}
            previous_g = g
        cm_open = float(book.price) if cm.bad[i] else cm.O[i]
        spot_open = previous_spot if spot.bad[k] else spot.O[k]
        event_boundary = decision is not None or np.isfinite(cm.funding_rate[i])
        if event_boundary:
            # The first account coin is created at actual Spot open. Do not attribute
            # a pre-existing coin position across the pre-window Spot opening gap.
            book.mark(time,cm_open,spot_open,'open')
            if np.isfinite(cm.funding_rate[i]):
                old = book.face
                flow = book.fund(time,float(cm.funding_rate[i]),cost) if not liquidated else 0.
                funding.append({'time':time,'rate':cm.funding_rate[i],'face_before':old,
                                'reference_price':cm_open,'coin_flow':flow,'bad':cm.bad[i],
                                'archive_gap':cm.archive_gap[i]})
            # Old holdings cannot escape opening pressure by voluntary rebalancing.
            opening_event = None
            if not liquidated and not cm.bad[i] and book.face:
                opening_event = pressure_event(float(book.coin),book.face,float(book.price),cm_open,cm_open,cm_open)
                if opening_event is not None:
                    pressure.append(book.liquidate(time,opening_event))
                    liquidated = True
            if decision is not None:
                status = 'liquidated' if liquidated else 'hold'
                if decision['triggered'] and not liquidated:
                    pending = False
                    if target == book.face:
                        status = 'zero_delta'
                    elif cm.bad[i]:
                        status,pending = 'canceled_bad',True
                    else:
                        orders.append(book.trade(time,target,cost))
                        status = 'executed'
                if liquidated:
                    pending = False
                decision.update(status=status,pending_after=pending,face_after=book.face)
                decisions.append(decision)
            if not liquidated and not cm.bad[i] and book.face:
                event = pressure_event(float(book.coin),book.face,float(book.price),cm_open,cm.H[i],cm.L[i])
                if event is not None:
                    pressure.append(book.liquidate(time,event))
                    liquidated = True
                    pending = False
        elif not liquidated and not cm.bad[i] and book.face:
            # In a normal holding minute the single close-to-close attribution
            # is exact; pressure still checks the actual opening before extrema.
            event = pressure_event(float(book.coin),book.face,float(book.price),cm_open,cm.H[i],cm.L[i])
            if event is not None:
                book.mark(time,float(book.price),spot_open,'pressure_spot_open')
                pressure.append(book.liquidate(time,event))
                liquidated = True
                pending = False
        cm_close = float(book.price) if cm.bad[i] else cm.C[i]
        spot_close = previous_spot if spot.bad[k] else spot.C[k]
        book.mark(time+_LAST_INSTANT,cm_close,spot_close,'close')
        held_face = book.face
        if terminal and book.face:
            orders.append(book.trade(time+_LAST_INSTANT,0.,cost,terminal=True))
        if book.coin < 0:
            error = ScopeError(f'negative coin account outside explicit pressure floor at {time}')
            error.partial = {'orders':orders,'funding':funding,'pressure_events':pressure,'attribution':book.rows}
            raise error
        spot_equity = book.coin*_LD(spot_close)
        cm_equity = book.coin*_LD(cm_close)
        cash_equity = spot_equity
        if terminal:
            cash_equity = book.coin*_LD(spot_close*(1-cost.sell_slip))*_LD(1-cost.fee)
            terminal_fx = cash_equity-spot_equity
        leverage = float(book.face/cm_equity) if cm_equity else 0.
        data[j] = (float(cash_equity),float(spot_equity),float(cm_equity),float(book.coin),
                   float(book.wallet),float(book.upnl()),book.face,float(cm_close),float(spot_close),
                   leverage,float(liquidated),held_face)
        previous_spot = spot_close
        book.assert_wallet()
    attr = book.rows
    def ld_sum(column):
        return np.sum(attr[column].to_numpy(dtype=np.longdouble),dtype=np.longdouble)
    spot_parts = initial_spot+terminal_fx+ld_sum('holding_spot_usd')+ld_sum('contract_mtm_spot_usd')+ld_sum('funding_spot_usd')+ld_sum('slippage_spot_usd')-ld_sum('fee_spot_usd')+ld_sum('zero_floor_spot_usd')
    cm_parts = initial_cm+ld_sum('holding_cm_usd')+ld_sum('contract_mtm_cm_usd')+ld_sum('funding_cm_usd')+ld_sum('slippage_cm_usd')-ld_sum('fee_cm_usd')+ld_sum('zero_floor_cm_usd')
    spot_error = float(cash_equity-_LD(INITIAL_CAPITAL)-spot_parts)
    cm_error = float(cm_equity-_LD(INITIAL_CAPITAL)-cm_parts)
    coin_error = float(book.coin-_LD(initial)-ld_sum('contract_mtm_coin')-ld_sum('funding_coin')-ld_sum('slippage_coin')+ld_sum('fee_coin')-ld_sum('zero_floor_coin'))
    if abs(spot_error) >= 1e-9 or abs(cm_error) >= 1e-9 or abs(coin_error) >= 1e-12:
        error = ScopeError('terminal inverse dynamic attribution failed')
        error.partial = {'spot_error':spot_error,'cm_error':cm_error,'coin_error':coin_error,
                         'orders':orders,'funding':funding,'attribution':attr,'pressure_events':pressure}
        raise error
    minute = pd.DataFrame(data,index=index,columns=['equity','equity_spot','equity_cm','coin_equity','wallet_coin',
                              'unrealized_pnl_coin','face','cm_price','spot_price','actual_leverage','liquidated','held_face'])
    minute['contract_leverage'] = minute['actual_leverage']
    minute['total_reference_exposure'] = np.where(minute['coin_equity'] > 0,1.+minute['contract_leverage'],0.)
    minute['actual_leverage'] = minute['total_reference_exposure']
    minute.loc[minute.index[-1],'actual_leverage'] = 0.  # Cash-closed primary valuation.
    minute['liquidated'] = minute['liquidated'].astype(bool)
    minute['bad'] = cm.bad[start:stop]|spot.bad[ss:se]
    minute['archive_gap'] = cm.archive_gap[start:stop]|spot.archive_gap[ss:se]
    de,dr = _daily(minute['equity'],days)
    metadata = {'version':version,'cost':cost.name,'asset':cm.asset,'window':window.name,
                'initial_capital':INITIAL_CAPITAL,'initial_coin':initial,'initial_coin_estimate':estimate,
                'final_coin':float(book.coin),'coin_growth':float(book.coin/_LD(initial)-1),
                'final_equity':float(cash_equity),'final_spot_equity_before_fx':float(spot_equity),
                'final_cm_equity':float(cm_equity),'initial_conversion_pnl':float(initial_spot),
                'terminal_conversion_pnl':float(terminal_fx),'initial_cm_bridge':float(initial_cm),
                'account_reconciliation_error':spot_error,'cm_account_reconciliation_error':cm_error,
                'coin_reconciliation_error':coin_error,'max_wallet_reconciliation_error':book.max_wallet_error,
                'order_count':len(orders),'adjustment_count':sum(not r['terminal'] for r in orders),
                'total_funding_coin':float(ld_sum('funding_coin')),
                'total_fee_coin':float(ld_sum('fee_coin')),'total_slippage_coin':float(ld_sum('slippage_coin')),
                'total_floor_credit_coin':float(ld_sum('zero_floor_coin')),
                'liquidated':liquidated,'whole_contracts':bool(whole_contracts),
                'pressure_model':'Fixed 0.5% MMR on contract 1m OHLC, residual coin absorbing trading state',
                'pressure_order':'freeze, mark old open, old funding, old opening pressure, net trade, new opening/extreme pressure'}
    return Result(minute,de,dr,pd.DataFrame(orders),pd.DataFrame(decisions),pd.DataFrame(funding),metadata,
                  attr,pd.DataFrame(pressure))

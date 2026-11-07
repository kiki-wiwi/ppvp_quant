"""Hand-sized accounting checks on output metrics, not strategy fitting."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest


def test_summary_reports_risk_and_costs_from_own_account():
    from trend_hold_study.reporting import portfolio_summary
    days=pd.date_range('2020-03-01',periods=2,tz='UTC')
    minute=pd.DataFrame({'equity':[110000.,99000.],'qty':[1.,0.],
        'mark_price':[110.,99.],'actual_leverage':[.001,0.]},index=days)
    result=SimpleNamespace(minute=minute,daily_returns=pd.Series([.10,-.10],index=days),
        metadata={'version':'T_daily_trend','cost':'main_17','total_fees':3.,'total_funding':-2.,
                  'order_count':2,'adjustment_count':1,'turnover_notional':200.,
                  'total_slippage_cost':1.,'max_actual_leverage':.001,
                  'minutes_above_leverage_cap':0,'maintenance_held_minutes':0,'gap_held_minutes':0})
    result.orders=pd.DataFrame(columns=['qty_before','qty_target'])
    result.decisions=pd.DataFrame({'trend_flip':[False,False]})
    row=portfolio_summary(result)
    assert row['total_return']==pytest.approx(-.01)
    assert row['max_drawdown']==pytest.approx(-.10)
    assert row['exposure_fraction']==.5
    assert row['fees']==3. and row['funding_cashflow']==-2.
    assert row['realized_volatility']==pytest.approx(np.std([.10,-.10],ddof=1)*np.sqrt(365))


def test_safe_json_records_undefined_without_invalid_json(tmp_path):
    from trend_hold_study.runtime import write_json
    p=tmp_path/'x.json'
    write_json(p,{'x':np.nan,'date':pd.Timestamp('2020-03-01',tz='UTC'),'flag':np.bool_(True)})
    import json
    got=json.loads(p.read_text())
    assert got['x'] is None and got['flag'] is True


def test_holding_intervals_are_not_counted_as_rebalances_and_terminal_is_separate():
    from trend_hold_study.reporting import position_counts,period_rows
    dates=pd.date_range('2020-03-01',periods=4,tz='UTC')
    minute=pd.DataFrame({'equity':[100000.]*4,'qty':[1.,2.,0.,0.],
        'held_qty':[1.,2.,0.,1.],'mark_price':[100.]*4,'actual_leverage':[.001,.002,0.,0.]},index=dates)
    orders=pd.DataFrame({'time':list(dates)+[dates[-1]+pd.Timedelta(days=1)-pd.Timedelta(nanoseconds=1)],
        'qty_before':[0.,1.,2.,0.,1.],'qty_target':[1.,2.,0.,1.,0.],
        'terminal':[False,False,False,False,True],'fee':[0.]*5,
        'turnover_notional':[100.]*5,'turnover_fraction':[.001]*5})
    decisions=pd.DataFrame({'time':dates,'g':[True,True,False,True],
        'trend_flip':[False,False,True,True]})
    funding=pd.DataFrame({'time':pd.Series([],dtype='datetime64[ns, UTC]'),'cash_flow':pd.Series([],dtype=float)})
    result=SimpleNamespace(minute=minute,orders=orders,decisions=decisions,funding=funding,
        daily_returns=pd.Series(0.,index=dates),daily_equity=pd.Series(100000.,index=dates),
        metadata={'version':'T_daily_trend','cost':'main_17'})
    counts=position_counts(result)
    assert counts=={'signal_switch_count':2,'position_entry_count':2,
        'completed_holding_intervals':2,'held_calendar_days':3}
    annual=next(r for r in period_rows(result) if r['period']=='2020')
    assert annual['order_count']==5 and annual['adjustment_count']==4
    assert annual['terminal_order_count']==1
    for key,value in counts.items():assert annual[key]==value

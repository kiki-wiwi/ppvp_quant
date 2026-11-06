import pandas as pd
import pytest

def test_unfilled_opportunity_contribution_uses_all_fathers():
    from dip_limit_study.reporting import paired_diagnostics
    events=pd.DataFrame({'M_net_bps':[10.,-20.,30.],'L_net_bps':[15.,0.,0.],
        'difference_bps':[5.,20.,-30.],'L_filled':[True,False,False],
        'L_status':['filled','expired','rejected'],'L_wait_minutes':[2.,60.,0.]})
    got=paired_diagnostics(events)
    assert got['father_count']==3 and got['fill_rate']==pytest.approx(1/3)
    assert got['filled_contribution_bps']==pytest.approx(5/3)
    assert got['unfilled_contribution_bps']==pytest.approx(-10/3)
    assert got['mean_difference_bps']==pytest.approx(-5/3)

def test_empty_event_diagnostic_reports_absence_not_fake_win():
    from dip_limit_study.reporting import paired_diagnostics
    got=paired_diagnostics(pd.DataFrame(columns=['M_net_bps','L_net_bps','difference_bps','L_filled','L_status','L_wait_minutes']))
    assert got['father_count']==0 and got['mean_difference_bps'] is None


def test_zero_trade_portfolio_reporting_preserves_undefined_expectancy():
    from dip_limit_study.reporting import comparison_rows
    from dip_limit_study.engine import run_portfolio
    from dip_limit_study.scope import make_context
    from test_dip_limit_scope import frame
    signals=pd.DataFrame(columns=['signal_time','decision_time','deadline','eref','atr','sl','limit','target','eligible','skip_reason'])
    result=run_portfolio(make_context(frame()),signals,'M_market','gross')
    table=comparison_rows({('M_market','gross',False):result},{})
    assert table.iloc[0].trades==0 and table.iloc[0].total_return==0
    assert pd.isna(table.iloc[0].net_bps)

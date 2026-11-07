"""Synthetic inference checks; production draw counts and seeds are never reduced."""
from hashlib import sha256
import math
import numpy as np
import pandas as pd
import pytest

from trend_hold_venue_study.config import WEBB_DRAWS, WEBB_SEED, MONTH_DRAWS, MONTH_SEED
from trend_hold_venue_study.statistics import analyze, _daily_returns, _holm, _diagnostic_gates


def test_full_registered_draws_pairing_identical_difference_and_descriptive_post_is():
    days = pd.date_range('2024-01-01','2024-04-01',inclusive='left',freq='D',tz='UTC').as_unit('ns')
    values = np.resize([-.004,.002,.001,.006,-.002],len(days))
    t = pd.Series(values,index=days)
    features = pd.DataFrame({'g': np.arange(len(days))%2 == 0},index=days)
    out = analyze(t,t.copy(),t-.0001,t-.0001,features,5)
    assert out['metadata']['calendar']['months'] == 3
    assert out['metadata']['calendar']['days'] == 91
    assert out['metadata']['family'] == ['T','D']
    assert out['metadata']['family_size'] == 2
    assert out['metadata']['automatic_promotion'] is False
    assert out['tests'].loc['D','p_raw'] == 1.
    assert 'identical_main_daily_returns' in out['tests'].loc['D','forced_p_one_reasons']
    assert out['intervals'].loc['D_daily_mean','lower'] == 0.
    assert out['intervals'].loc['D_daily_mean','upper'] == 0.
    assert out['intervals'].loc['D_daily_mean','valid_resamples'] == MONTH_DRAWS
    assert out['gates']['passed'].isna().all()
    assert out['metadata']['calendar']['half_split'] is None
    points = np.array([-math.sqrt(1.5),-1.,-math.sqrt(.5),math.sqrt(.5),1.,math.sqrt(1.5)])
    expected_weights = np.random.Generator(np.random.PCG64(WEBB_SEED)).choice(points,size=(WEBB_DRAWS,3))
    expected_draws = np.random.Generator(np.random.PCG64(MONTH_SEED)).integers(0,3,size=(MONTH_DRAWS,3))
    np.testing.assert_array_equal(out['matrices']['webb_weights'],expected_weights)
    np.testing.assert_array_equal(out['matrices']['month_indices'],expected_draws)
    assert out['metadata']['matrices']['webb_weights']['sha256'] == sha256(expected_weights.tobytes()).hexdigest()
    assert out['metadata']['matrices']['month_indices']['sha256'] == sha256(expected_draws.tobytes()).hexdigest()
    assert out['tests'].loc['T','bootstrap_replicates'] == 99999
    assert out['tests'].loc['T','finite_mc_denominator'] == 100000


def test_holm_ties_use_T_first_and_keep_two_member_family():
    out = _holm({'D':.01,'T':.01})
    assert out['T']['rank'] == 1
    assert out['D']['rank'] == 2
    assert out['T']['p_holm'] == out['D']['p_holm'] == .02
    with pytest.raises(ValueError): _holm({'T':.01})


def test_zero_after_bankruptcy_return_convention_and_no_resurrection():
    days = pd.date_range('2024-01-01',periods=4,tz='UTC').as_unit('ns')
    values,total = _daily_returns(pd.Series([.1,-1.,0.,0.],index=days),days,'synthetic')
    assert values.tolist() == [.1,-1.,0.,0.]
    assert total == -1.
    with pytest.raises(ValueError,match='zero'):
        _daily_returns(pd.Series([.1,-1.,.01,0.],index=days),days,'synthetic')
    with pytest.raises(ValueError,match='below'):
        _daily_returns(pd.Series([.1,-1.01,0.,0.],index=days),days,'synthetic')


def test_calendar_mismatch_partial_month_and_duplicate_days_stop_before_draws():
    days = pd.date_range('2020-03-01','2020-06-01',inclusive='left',tz='UTC').as_unit('ns')
    t = pd.Series(0.,index=days)
    f = pd.DataFrame({'g':False},index=days)
    with pytest.raises(ValueError,match='calendar'):
        analyze(t,t.iloc[::-1],t,t,f,1)
    with pytest.raises(ValueError,match='complete'):
        analyze(t.iloc[1:],t.iloc[1:],t.iloc[1:],t.iloc[1:],f.iloc[1:],1)
    duplicate = pd.concat([t.iloc[:1],t])
    with pytest.raises(ValueError,match='calendar'):
        analyze(duplicate,duplicate,duplicate,duplicate,f,1)


def test_IS_gates_use_fixed_2022_split_and_stable_best_three_months():
    days = pd.date_range('2020-03-01','2024-01-01',inclusive='left',tz='UTC').as_unit('ns')
    # All monthly paired sums are exactly zero, so chronological ties must win.
    daily = pd.DataFrame({'T':.001,'P':.001,'D':0.,'T_harsh':.0005,
                          'P_harsh':.0005,'D_harsh':0.},index=days)
    states = pd.Series(np.arange(len(days))%2 == 0,index=days)
    monthly = daily.groupby(days.strftime('%Y-%m'),sort=True).agg(day_count=('T','size'),D_sum=('D','sum'))
    tests = pd.DataFrame({'p_holm':[.01,1.]},index=['T','D'])
    cumulative = {'main':{'T':.5,'P':.5},'harsh':{'T':.2,'P':.2}}
    gates = _diagnostic_gates(daily,states,cumulative,monthly,tests)
    assert bool(gates.loc['V1','passed'])
    assert not bool(gates.loc['V3','passed'])
    assert gates.loc['V5','details']['removed_months'] == ['2020-03','2020-04','2020-05']
    counts = gates.loc['V1','details']['state_counts_by_half']
    assert sum(counts[0].values()) == int((days < pd.Timestamp('2022-01-01',tz='UTC')).sum())
    assert sum(counts[1].values()) == int((days >= pd.Timestamp('2022-01-01',tz='UTC')).sum())

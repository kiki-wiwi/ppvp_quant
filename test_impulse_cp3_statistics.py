"""Synthetic-only checks for the frozen CP3 statistical implementation."""
import numpy as np
import pandas as pd
import pytest
from impulse_study import cp3_statistics as S
from impulse_study.scope import ScopeError
from ppvp_audit.nulls import cluster_bootstrap_mean


def ledger(values, months=None):
    values = np.asarray(values, float)
    if months is None:
        months = pd.date_range('2020-03-01', periods=len(values), freq='MS', tz='UTC')
    entry = pd.DatetimeIndex(months) + pd.Timedelta(minutes=1)
    return pd.DataFrame({'trigger_time': entry - pd.Timedelta(minutes=1), 'entry_time': entry,
                         'exit_time': entry + pd.Timedelta(minutes=15), 'net_bps': values})


def returns(values=None):
    idx = pd.date_range('2020-03-01', '2023-12-31', freq='D', tz='UTC')
    if values is None:
        values = .0003 + .01 * np.sin(np.arange(len(idx)) * .371)
    return pd.Series(values, index=idx)


def test_scope_rejects_val_and_prewindow_trades():
    for bad in ('2024-01-01', '2020-02-29'):
        tr = ledger([1.0], pd.DatetimeIndex([bad], tz='UTC'))
        with pytest.raises(ScopeError):
            S.cluster_mean(tr, ci_seed=1, p_seed=2)
    bad = returns()
    bad.index = bad.index + pd.Timedelta(days=1)
    with pytest.raises(ScopeError):
        S.month_block_sharpe(bad, seed=3)


def test_cluster_percentiles_match_frozen_helper_and_unequal_month_weights():
    dates = np.repeat(pd.date_range('2020-03-01', periods=12, freq='MS', tz='UTC'), np.arange(1, 13))
    x = np.arange(len(dates), dtype=float) / 9 - 2
    tr = ledger(x, dates)
    got = S.cluster_mean(tr, ci_seed=71, p_seed=None)
    ref = cluster_bootstrap_mean(x, pd.DatetimeIndex(tr.entry_time).strftime('%Y-%m'), 10000, seed=71)
    for k in ('lo','hi','p_le_0'):
        assert got[k] == ref[k]
    assert got['mean'] == pytest.approx(x.mean(), rel=1e-12)
    assert got['n_clusters'] == 12


def test_webb_single_literal_frozen_draw_order_and_zero_se_replicates():
    x = np.array([-3.,1.,4.,5.,-2.,8.,2.,-7.,3.,1.,6.,9.])
    got = S.cluster_mean(ledger(x), ci_seed=1, p_seed=9001)
    w = np.random.default_rng(9001).choice(S.WEBB, size=(99999, 12))
    star = w * x
    m = star.sum(axis=1) / 12
    se = np.sqrt(12/11 * ((star - m[:,None])**2).sum(axis=1) / 12**2)
    tstar = np.divide(m, se, out=np.zeros(len(m)), where=se > 0)
    se0 = np.sqrt(12/11 * ((x-x.mean())**2).sum()/12**2)
    assert got['t'] == pytest.approx(x.mean()/se0)
    assert got['p'] == np.mean(tstar >= got['t'])
    assert got['mcse_positive'] == pytest.approx(np.sqrt(got['p']*(1-got['p'])/99999))


def test_month_threshold_and_degenerate_samples_enter_holm_as_one():
    short = S.cluster_mean(ledger(np.arange(11)), ci_seed=1, p_seed=2)
    assert short['p'] == 1 and short['lo'] is None and short['t'] is None
    constant = S.cluster_mean(ledger(np.ones(12)), ci_seed=1, p_seed=2)
    assert constant['p'] == 1 and constant['lo'] == 1 and constant['t'] is None
    p = {name: 1.0 for name in S.NAMES}
    p[S.NAMES[0]] = .003
    family = S.holm_family(p, .025)
    assert family['size'] == 10
    assert family['tests'][S.NAMES[0]]['adjusted_p'] == .03
    assert not family['tests'][S.NAMES[0]]['passed']


def test_paired_webb_common_weights_and_shared_union_months():
    dates = pd.date_range('2020-03-01', periods=14, freq='MS', tz='UTC')
    a = ledger(np.arange(12)/3 + .5, dates[:12])
    b = ledger(np.arange(12)/7 - .2, dates[2:])
    got = S.paired_cluster_mean(a, b, ci_seed=77, p_seed=88)
    ka = np.r_[np.ones(12), 0.,0.]
    kb = np.r_[0.,0.,np.ones(12)]
    sa = np.r_[a.net_bps.to_numpy(), 0.,0.]
    sb = np.r_[0.,0.,b.net_bps.to_numpy()]
    mu = (sa.sum()+sb.sum())/24
    w = np.random.default_rng(88).choice(S.WEBB,size=(99999,14))
    astar = mu*ka+w*(sa-mu*ka)
    bstar = mu*kb+w*(sb-mu*kb)
    ma,mb = astar.sum(axis=1)/12,bstar.sum(axis=1)/12
    phi = (astar-ma[:,None]*ka)/12-(bstar-mb[:,None]*kb)/12
    sd = np.sqrt(14/13*(phi**2).sum(axis=1))
    t = np.divide(ma-mb,sd,out=np.zeros(len(sd)),where=sd>0)
    assert got['p_positive'] == np.mean(t>=got['t'])
    assert got['p_negative'] == np.mean(t<=got['t'])
    assert got['n_clusters'] == 14
    draws = np.random.default_rng(77).integers(0,14,size=(10000,14))
    na,nb = ka[draws].sum(axis=1),kb[draws].sum(axis=1)
    ok = (na>0)&(nb>0)
    d = sa[draws].sum(axis=1)[ok]/na[ok]-sb[draws].sum(axis=1)[ok]/nb[ok]
    assert got['lo'] == np.percentile(d,2.5)
    assert got['discarded'] == int((~ok).sum())


def test_month_sharpe_uses_full_natural_blocks_not_equal_day_weights():
    r = returns()
    got = S.month_block_sharpe(r,seed=42)
    groups = [x.to_numpy() for _,x in r.groupby(r.index.strftime('%Y-%m'))]
    draws = np.random.default_rng(42).integers(0,46,size=(5000,46))
    sample = np.array([S.annualized_sharpe(np.concatenate([groups[i] for i in row])) for row in draws])
    assert got['lo'] == pytest.approx(np.percentile(sample,2.5),abs=1e-12)
    assert got['hi'] == pytest.approx(np.percentile(sample,97.5),abs=1e-12)
    assert got['p0'] == np.mean(sample <= 0)
    assert got['n_months'] == 46 and got['n_days'] == 1401


def test_paired_sharpe_identical_paths_tie_and_constant_degeneracy():
    r = returns()
    got = S.paired_month_block_sharpe(r,r,seed=27)
    assert got['diff'] == 0 and got['lo'] == 0 and got['hi'] == 0
    assert got['p_two_sided'] == 1 and got['p_positive'] == 1
    zero = returns(np.zeros(1401))
    bad = S.month_block_sharpe(zero,seed=27)
    assert bad['sharpe'] is None and bad['discarded'] == 5000 and bad['p0'] == 1


def test_seed_table_offsets_and_sorted_f5_pair():
    assert S.seed_for('g2_ci',0)==20261306
    assert S.seed_for('g3_p',9)==20261465
    assert S.seed_for('f5',9,other=2)==20261106+700+29
    assert S.seed_for('g2_p',4,sensitivity=3)==20261106+30254
    assert S.seed_for('g4',4,auxiliary=True)==20261106+5504
    assert S.seed_for('appendix_p',5)==20261106+3555


def test_point_gates_halves_empty_and_tied_top_months():
    tr = ledger(np.ones(12))
    detail = S.point_gate_statistics(tr,tr,None)
    assert not detail['G5']
    assert detail['removed_top_months']==['2020-03','2020-04','2020-05']
    assert detail['G7']
    empty = tr.iloc[:0]
    d = S.point_gate_statistics(empty,empty,empty)
    assert not any(d[k] for k in ['G1','G5','G6','G7','G8z'])


def test_box_decision_opposite_directions_has_priority():
    pairs = [{'a':'A1','positive':True,'negative':False,'half_differences':[1.,1.], 'a_passes_g2':True},
             {'a':'A2','positive':False,'negative':True,'half_differences':[-1.,-1.], 'a_passes_g2':False}]
    d=S.box_decision(pairs)
    assert d['rule']==1
    pairs[1]['negative']=False
    assert S.box_decision(pairs)['rule']==2
    pairs[0]['a_passes_g2']=False
    d=S.box_decision(pairs)
    assert d['rule']==3 and d['unmet_conditions']


def test_cost_sensitive_candidate_and_g8z_requirements():
    gates={name:{f'G{k}':False for k in range(1,8)} for name in S.NAMES}
    n=S.NAMES[4]
    gates[n]={f'G{k}':True for k in range(1,8)}
    gates[n]['G6']=False
    gates[n]['G8z']=False
    assert S.eligible_sets(gates)==([],[])
    gates[n]['G8z']=True
    assert S.eligible_sets(gates)==([],[n])


def rich_result(values, daily=None):
    from types import SimpleNamespace
    values=np.asarray(values,float)
    times=pd.date_range('2020-03-01','2023-12-30',periods=len(values),tz='UTC').floor('min')
    tr=ledger(values,times)
    tr['trigger_idx']=np.arange(len(tr))
    tr['gross_bps']=values+17
    tr['net_R']=values/100
    tr['gross_pnl']=(values+17)*10
    tr['net_pnl']=values*10
    tr['risk_amount']=1000.
    tr['reason_name']='time_bar1'
    tr['reason']=4
    tr['minutes_held']=15
    tr['funding']=0.
    tr['cap_bound']=False
    tr['floor_bound']=False
    daily=returns() if daily is None else daily
    eq=S.VS.INITIAL_CAPITAL*(1+daily).cumprod()
    return SimpleNamespace(trades=tr,daily_equity=eq,equity=None)


def test_paired_metrics_share_one_generator_per_registered_seed(monkeypatch):
    a=ledger(np.arange(12.)+.5)
    b=ledger(np.arange(12.)/3-.5)
    for tr in (a,b):
        tr['gross_bps']=tr.net_bps+17
        tr['net_R']=tr.net_bps/100
    original=np.random.default_rng
    calls=[]
    def counted(seed):
        calls.append(seed)
        return original(seed)
    monkeypatch.setattr(S.np.random,'default_rng',counted)
    got=S.paired_cluster_metrics(a,b,ci_seed=77,p_seed=88)
    assert calls==[77,88]
    assert list(got)==['net_bps','gross_bps','net_R']
    assert got['net_bps']['lo']==pytest.approx(got['gross_bps']['lo'])
    assert got['net_R']['mean']==pytest.approx(got['net_bps']['mean']/100)


def test_stationary_auxiliary_matches_frozen_public_helpers_and_one_rng(monkeypatch):
    import attribution as A
    a,b=returns(),returns(.0001+.008*np.cos(np.arange(1401)*.28))
    ref=A.stationary_bootstrap_sharpe(a,n_boot=5000,mean_block=10.,seed=6107)
    pair_ref=A.paired_sharpe_diff(a,b,n_boot=5000,mean_block=10.,seed=6108,benchmark_factor=1.)
    original=np.random.default_rng
    calls=[]
    def counted(seed):
        calls.append(seed)
        return original(seed)
    monkeypatch.setattr(A.np.random,'default_rng',counted)
    got=S.stationary_sharpe(a,seed=6107)
    pair=S.paired_stationary_sharpe(a,b,seed=6108)
    assert calls==[6107,6108]
    for key in ('sharpe','lo','hi','p_gt0'):
        assert got[key]==ref[key]
    for key in ('diff','lo','hi','p_gt0','benchmark','benchmark_factor'):
        assert pair[key]==pair_ref[key]
    assert got['discarded']==pair['discarded']==0
    assert got['auxiliary_only'] and pair['auxiliary_only']


def test_stationary_degenerate_is_defined_as_p_one_without_changing_primary():
    zero=returns(np.zeros(1401))
    one=S.stationary_sharpe(zero,seed=601)
    pair=S.paired_stationary_sharpe(zero,returns(),seed=602)
    assert one['sharpe'] is None and one['p']==1 and one['discarded']==5000
    assert pair['diff'] is None and pair['p']==1 and pair['discarded']==5000


def test_complete_decision_pipeline_fixed_families_cost_candidate_and_ties():
    from types import SimpleNamespace
    r=returns(.01+.001*np.sin(np.arange(1401)*.3))
    x=50+np.sin(np.arange(240))
    main={name:rich_result(x,r) for name in S.NAMES}
    harsh={name:rich_result(x-17,r) for name in S.NAMES}
    harsh[S.NAMES[0]]=rich_result(-np.ones(240),r)
    excess={name:result.trades.assign(excess_bps=10+np.cos(np.arange(240))) for name,result in main.items()}
    causal={name:rich_result(x,r) for name in (S.NAMES[4],S.NAMES[9])}
    out=S.evaluate_study(main,harsh,excess,causal,context=SimpleNamespace(box_ok=np.arange(240)%2==0))
    assert out['families']['F1']['size']==out['families']['F2']['size']==10
    assert out['families']['F3_positive']['size']==out['families']['F3_negative']['size']==2
    assert out['families']['F4']['size']==4
    first=out['versions'][S.NAMES[0]]['gates']
    assert all(first[k] for k in ['G1','G2','G3','G4','G5','G7']) and not first['G6']
    qualified=out['decisions']['qualification']
    assert qualified['cost_sensitive_versions']==[S.NAMES[0]]
    assert len(qualified['eligible_versions'])==9 and qualified['winner'] is None
    assert out['decisions']['pure_alpha']['versions']==S.NAMES[5:]
    assert all(item['best_version'] is None for item in out['decisions']['stops'].values())
    assert out['versions'][S.NAMES[4]]['gates']['G8z'] and out['versions'][S.NAMES[9]]['gates']['G8z']


def test_cp4_hook_keeps_frozen_seed_shifts_and_does_not_mutate_payload(monkeypatch):
    calls=[]
    def descriptive(results,section):
        calls.append(('descriptive',section,results))
        return {'recorded':section}
    monkeypatch.setattr(S,'_sensitivity_descriptive_results',descriptive)
    hook=S.make_cp4_statistics_hook({})
    payload={'section':7,'results':{'a':object()},'context':None}
    out=hook(payload)
    assert out['primary_gates_unchanged']
    assert calls[0][1]==7 and calls[0][2] is payload['results']
    assert set(payload)=={'section','results','context'}
    with pytest.raises(ValueError):
        hook({'section':10,'results':{}})


def test_unique_winner_uses_two_sided_point025_and_nine_f5_pairs():
    from types import SimpleNamespace
    x=60+np.sin(np.arange(240))
    main={name:rich_result(x,returns(.01+v*.001+.001*np.sin(np.arange(1401)*.3)))
          for v,name in enumerate(S.NAMES)}
    harsh={name:rich_result(x-17) for name in S.NAMES}
    excess={name:result.trades.assign(excess_bps=10+np.cos(np.arange(240))) for name,result in main.items()}
    causal={name:rich_result(x) for name in (S.NAMES[4],S.NAMES[9])}
    out=S.evaluate_study(main,harsh,excess,causal,context=SimpleNamespace(box_ok=np.arange(240)%2==0))
    qualification=out['decisions']['qualification']
    assert qualification['eligible_versions']==S.NAMES
    assert qualification['candidate']==qualification['winner']==S.NAMES[9]
    assert out['families']['F5']['size']==9 and out['families']['F5']['level']==.025
    for v,name in enumerate(S.NAMES[:-1]):
        pair=out['winner_comparisons'][S._pair_key(S.NAMES[9],name)]
        assert pair['diff']>0 and pair['p_two_sided']==0
        assert pair['seed']==S.VS.SEED_BASE+700+10*v+9


def test_empty_versions_remain_in_families_and_fail_all_gates():
    from types import SimpleNamespace
    empty=rich_result([] ,returns(np.zeros(1401)))
    main={name:empty for name in S.NAMES}
    excess={name:empty.trades.assign(excess_bps=pd.Series(dtype=float)) for name in S.NAMES}
    causal={name:empty for name in (S.NAMES[4],S.NAMES[9])}
    out=S.evaluate_study(main,main,excess,causal,context=SimpleNamespace(box_ok=np.zeros(0,bool)))
    assert out['families']['F1']['size']==out['families']['F2']['size']==10
    assert all(t['raw_p']==1 for family in ('F1','F2') for t in out['families'][family]['tests'].values())
    assert all(not any(detail['gates'].values()) for detail in out['versions'].values())
    assert not out['decisions']['qualification']['eligible_versions']

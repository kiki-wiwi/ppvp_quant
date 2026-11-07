"""Synthetic-only coverage of the four CP2 caller-data charts."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from trend_hold_venue_study.plotting import FIGURES, render_outputs


COSTS = ('gross_0', 'main_17', 'harsh_34')
VERSIONS = ('V1_USDM', 'P_USDM', 'V2_SPOT', 'P_SPOT',
            'V3_CM_DeltaMatched', 'P_CM_DeltaMatched', 'V4_CM_CoinAccum', 'HODL_1x_Coin')


def fixture():
    accounts, rows = {}, []
    for window, versions in [('IS-Full', VERSIONS[:4]), ('IS-Common', VERSIONS)]:
        for position, version in enumerate(versions):
            for c, cost in enumerate(COSTS):
                job = {'window': window, 'version': version, 'cost': cost,
                       'signal_source': 'UM', 'whole_contracts': False}
                job_id = '__'.join((window, version, cost))
                dates = pd.date_range('2020-03-01' if window == 'IS-Full' else '2020-09-01',
                                      periods=12, freq='MS', tz='UTC').as_unit('ns')
                ret = .1 + position * .015 - c * .01
                daily = pd.DataFrame({'equity': 100_000. * (1. + np.linspace(0., ret, len(dates))),
                                      'return': np.zeros(len(dates))}, index=dates)
                metadata = {'initial_coin': 2., 'version': version, 'cost': cost,
                            'window': window, 'source': 'synthetic'}
                if position >= 4:
                    daily['coin_equity'] = 2. * (1. + np.linspace(0., .01-c*.002, len(dates)))
                    daily['equity_spot'] = daily['equity'] + 10.
                    daily['equity_cm'] = daily['equity'] + 20.
                accounts[job_id] = {'daily': daily, 'metadata': metadata, 'job': job}
                rows.append({**job, 'job_id': job_id, 'total_return': ret,
                             'coin_growth': .01-c*.002 if position >= 4 else np.nan,
                             'funding_usd': -100. if version.endswith('USDM') else 30. if 'CM' in version else 0.,
                             'fee_usd': c*20., 'slippage_usd': -c*10., 'conversion_usd': -c*5.,
                             'max_drawdown': .05+position*.01, 'cash_clip_count': int('SPOT' in version)})
    # Diagnostic rows exist in the batch but never enter the primary cost panels.
    for version in ('V3_CM_DeltaMatched', 'P_CM_DeltaMatched', 'V4_CM_CoinAccum'):
        primary = next(r for r in rows if r['window']=='IS-Common' and r['version']==version and r['cost']=='main_17')
        rows.append({**primary, 'job_id': primary['job_id']+'__whole_contracts', 'whole_contracts': True})
        rows.append({**primary, 'job_id': 'IS-CM-SelfSignal__'+version+'__main_17',
                     'window': 'IS-CM-SelfSignal', 'signal_source': 'CM'})
    families = ('IS-Full__UM', 'IS-Full__SPOT', 'IS-Common__UM',
                'IS-Common__SPOT', 'IS-Common__CM_DeltaMatched')
    stats = {family: {'gates': pd.DataFrame({'passed': [True, True, False, True, False, True]},
                                           index=['V1', 'V2', 'V3', 'V4', 'V5', 'V6'])}
             for family in families}
    dates = pd.date_range('2020-11-01', periods=90, freq='D', tz='UTC').as_unit('ns')
    self_signal = pd.DataFrame({'g_UM': np.arange(90)%2==0, 'g_CM': np.arange(90)%3==0,
                               'budget_UM': .4, 'budget_CM': .45}, index=dates)
    diagnostics = {'self_signal': self_signal,
                   'self_signal_summary': {'concordance_pct': float((self_signal.g_UM==self_signal.g_CM).mean()*100.)},
                   'rounding': pd.DataFrame({'version': ['V3_CM_DeltaMatched', 'P_CM_DeltaMatched', 'V4_CM_CoinAccum'],
                       'continuous_final_equity': [110_000., 120_000., 130_000.],
                       'whole_final_equity': [110_002., 119_998., 130_001.],
                       'continuous_coin_growth': [.01, .02, .03], 'whole_coin_growth': [.0101, .0199, .0301]}),
                   'crossvenue': pd.DataFrame()}
    return accounts, {'UM': pd.DataFrame(), 'CM': pd.DataFrame()}, stats, pd.DataFrame(rows), diagnostics


def test_four_figures_are_readable_and_do_not_mutate_caller_tables(tmp_path):
    import matplotlib.image as mpimg
    accounts, features, stats, comparison, diagnostics = fixture()
    before = deepcopy((accounts, comparison, diagnostics))
    paths = render_outputs(accounts, features, stats, comparison, diagnostics, tmp_path)
    assert paths == list(FIGURES)
    for name in paths:
        image = mpimg.imread(tmp_path/name)
        assert image.shape[0] >= 1000 and image.shape[1] >= 1500
        assert (tmp_path/name).stat().st_size > 10_000
    caption = (tmp_path/'figure_captions.md').read_text()
    assert 'additional collateral exposure' in caption
    assert 'funding is disabled' in caption
    assert 'within one venue/window' in caption
    assert 'separate initial accounts' in caption
    for key in accounts:
        pd.testing.assert_frame_equal(accounts[key]['daily'], before[0][key]['daily'])
        assert accounts[key]['metadata'] == before[0][key]['metadata']
    pd.testing.assert_frame_equal(comparison, before[1])
    pd.testing.assert_frame_equal(diagnostics['self_signal'], before[2]['self_signal'])


def test_missing_primary_account_stops_before_writing(tmp_path):
    accounts, features, stats, comparison, diagnostics = fixture()
    del accounts['IS-Common__V3_CM_DeltaMatched__main_17']
    with pytest.raises(ValueError, match='account'):
        render_outputs(accounts, features, stats, comparison, diagnostics, tmp_path)
    assert not list(tmp_path.iterdir())


def test_existing_output_is_never_overwritten(tmp_path):
    path = tmp_path/FIGURES[0]
    path.write_bytes(b'preserve existing result')
    with pytest.raises(FileExistsError):
        render_outputs(*fixture(), tmp_path)
    assert path.read_bytes() == b'preserve existing result'

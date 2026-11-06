"""Reporting gates must retain the inference result, without selecting anew."""
import pandas as pd


def test_comparison_includes_registered_gates_and_adjusted_p_values():
    from breakout_study.reporting import annotate_comparison
    table=pd.DataFrame([{'version':'P_channel','cost':'main_17'},
                        {'version':'P_channel','cost':'harsh_34'},
                        {'version':'V_vah','cost':'main_17'}])
    stats={'gates':{v:{f'N{i}':{'passed':i<5} for i in range(1,7)} for v in ('P_channel','V_vah')},
        'tests':{'P':{'p_raw':.01,'p_holm':.03},'V':{'p_raw':.4,'p_holm':.8},'D':{'p_raw':.1,'p_holm':.2}},
        'intervals':{'daily_mean':{v:{'lower':-.1,'upper':.2} for v in ('P_channel','V_vah')}},
        'conclusions':{'versions':{'P_channel':'cost_sensitive_candidate','V_vah':'not_candidate'}}}
    got=annotate_comparison(table,stats)
    assert got.loc[0,'p_holm']==.03 and got.loc[2,'p_holm']==.8
    assert got.loc[0,'N1'] and not got.loc[0,'N5']
    assert pd.isna(got.loc[1,'p_holm'])
    assert got.loc[0,'candidate_status']=='cost_sensitive_candidate'


def test_frozen_verification_rejects_a_changed_legacy_source_without_editing_it(monkeypatch):
    from pathlib import Path
    import pytest
    from breakout_study.data import verify_registered_assets
    original=Path.read_bytes
    def changed(path):
        raw=original(path)
        return raw+b'\n' if path.name=='ppvp_engine.py' else raw
    monkeypatch.setattr(Path,'read_bytes',changed)
    with pytest.raises(ValueError,match='frozen file mismatch: ppvp_engine.py'):
        verify_registered_assets()


def test_new_code_registration_rejects_changes_and_allows_only_scope_unlock(tmp_path):
    import hashlib,json,pytest
    from breakout_study.data import verify_implementation_registration
    source=tmp_path/'breakout_study';source.mkdir()
    (source/'scope.py').write_text('REAL_DATA_UNLOCKED = False\n')
    (source/'config.py').write_text('RISK_FRACTION = .005\n')
    out=tmp_path/'reports/channel_vs_vah_round';out.mkdir(parents=True)
    paths=('breakout_study/scope.py','breakout_study/config.py')
    hashes={p:hashlib.sha256((tmp_path/p).read_bytes()).hexdigest() for p in paths}
    (out/'implementation_hashes.json').write_text(json.dumps({'sha256':hashes}))
    verify_implementation_registration(tmp_path)
    (source/'scope.py').write_text('REAL_DATA_UNLOCKED = True\n')
    verify_implementation_registration(tmp_path)
    (source/'config.py').write_text('RISK_FRACTION = .05\n')
    with pytest.raises(ValueError,match='new implementation hash mismatch'):
        verify_implementation_registration(tmp_path)

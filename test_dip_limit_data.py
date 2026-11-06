"""No market payloads: lock ordering and registered source byte checks."""
import hashlib
import json
from pathlib import Path
import pytest


def test_new_loader_rejects_before_any_input_io():
    from dip_limit_study.data import load_real_context, load_perp_is, load_funding_is
    from dip_limit_study.scope import RealDataLockedError
    for fn in (load_real_context,load_perp_is,load_funding_is):
        with pytest.raises(RealDataLockedError): fn(Path('/path/that/must/not/be/read'))


def test_protected_source_changed_bytes_fail_closed(tmp_path):
    from dip_limit_study.data import verify_registered_assets
    folder=tmp_path/'reports/trend_dip_limit_round'; folder.mkdir(parents=True)
    (tmp_path/'old.py').write_bytes(b'original\n')
    reg={'protected_sources_sha256':{'old.py':hashlib.sha256(b'original\n').hexdigest()},
         'approved_plan_sha256':hashlib.sha256(b'approved\n').hexdigest()}
    (folder/'registration.json').write_text(json.dumps(reg))
    (folder/'approved_plan.md').write_bytes(b'approved\n')
    assert verify_registered_assets(tmp_path)['protected_sources_verified']==1
    (tmp_path/'old.py').write_bytes(b'changed\n')
    with pytest.raises(ValueError,match='protected source'): verify_registered_assets(tmp_path)


def test_registered_approved_snapshot_matches_external_approval():
    from dip_limit_study.data import verify_registered_assets
    reg=verify_registered_assets()
    assert reg['approved_plan_sha256']=='df36ec579c750ef7580b828148d9ead00c22d69c4eea3ed073036b5730c671a3'
    assert reg['protected_sources_verified']>=50


def test_new_hash_registration_allows_only_separate_future_unlock(tmp_path):
    from dip_limit_study.data import verify_implementation_registration
    source=tmp_path/'dip_limit_study';source.mkdir()
    source.joinpath('scope.py').write_text('REAL_DATA_UNLOCKED = False\n')
    source.joinpath('config.py').write_text('RISK_FRACTION=.005\n')
    names=['dip_limit_study/scope.py','dip_limit_study/config.py']
    doc={'sha256':{n:hashlib.sha256((tmp_path/n).read_bytes()).hexdigest() for n in names}}
    folder=tmp_path/'reports/trend_dip_limit_round';folder.mkdir(parents=True)
    (folder/'implementation_hashes.json').write_text(json.dumps(doc))
    verify_implementation_registration(tmp_path)
    source.joinpath('scope.py').write_text('REAL_DATA_UNLOCKED = True\n')
    verify_implementation_registration(tmp_path)
    source.joinpath('config.py').write_text('RISK_FRACTION=.05\n')
    with pytest.raises(ValueError,match='new implementation hash mismatch'):
        verify_implementation_registration(tmp_path)

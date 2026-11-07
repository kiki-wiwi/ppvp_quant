"""Synthetic source-registration validation for the explicitly approved revision."""
from copy import deepcopy
import pytest
from trend_hold_venue_study import cp2_control as c


def registration():
    original={'sha256':{name:'a'*64 for name in c.APPROVED_MODIFIED}}
    original['sha256']['immutable_historical.py']='c'*64
    revision={'baseline_commit':c.ORIGINAL_UNLOCK,'original_implementation_commit':c.ORIGINAL_IMPL,
        'modified_files_sha256':{name:{'before':'a'*64,'after':'b'*64} for name in c.APPROVED_MODIFIED},
        'added_files_sha256':{c.REVISION+'/notes.md':'d'*64}}
    return original,revision


def test_revision_binds_old_hashes_and_keeps_unmodified_entries():
    original,revision=registration()
    result=c.validate_revision(revision,original)
    assert result['immutable_historical.py']=='c'*64
    assert all(result[name]=='b'*64 for name in c.APPROVED_MODIFIED)
    assert original['sha256']['trend_hold_venue_study/data.py']=='a'*64


@pytest.mark.parametrize('case',['baseline','prior_hash','unapproved_code','outside_addition','self_reference'])
def test_revision_rejects_scope_expansion_or_rewriting_original_registration(case):
    original,revision=registration();revision=deepcopy(revision)
    if case=='baseline':revision['baseline_commit']='0'*40
    elif case=='prior_hash':revision['modified_files_sha256']['trend_hold_venue_study/data.py']['before']='e'*64
    elif case=='unapproved_code':revision['modified_files_sha256']['trend_hold_venue_study/portfolios.py']={'before':'a'*64,'after':'b'*64}
    elif case=='outside_addition':revision['added_files_sha256']['trend_hold_venue_study/new_strategy.py']='d'*64
    else:revision['added_files_sha256'][c.REVISION_MANIFEST]='e'*64
    with pytest.raises(ValueError):c.validate_revision(revision,original)

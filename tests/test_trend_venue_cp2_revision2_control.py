"""Synthetic-only second input revision and preserved commit-chain validation."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from trend_hold_venue_study import cp2_control as c, scope


def revision2_fixture():
    prior={name:'b'*64 for name in c.APPROVED_MODIFIED}
    prior[c.REVISION+'/notes.md']='c'*64
    prior['immutable.py']='d'*64
    revision={'baseline_commit':c.PRIOR_UNLOCK,'prior_implementation_commit':c.PRIOR_IMPL,
        'modified_files_sha256':{name:{'before':'b'*64,'after':'e'*64} for name in c.APPROVED_MODIFIED},
        'added_files_sha256':{c.REVISION2+'/notes.md':'f'*64,
                             'trend_hold_venue_study/input_binding.py':'a'*64}}
    return prior,revision


def test_revision2_binds_prior_effective_hashes_and_preserves_all_old_entries():
    prior,revision=revision2_fixture()
    effective=c.validate_revision2(revision,prior)
    assert effective['immutable.py']=='d'*64
    assert effective[c.REVISION+'/notes.md']=='c'*64
    assert all(effective[name]=='e'*64 for name in c.APPROVED_MODIFIED)
    assert all(prior[name]=='b'*64 for name in c.APPROVED_MODIFIED)
    assert effective['trend_hold_venue_study/input_binding.py']=='a'*64


@pytest.mark.parametrize('case',['baseline','prior_commit','original_not_prior_hash','modified_scope',
                                'unapproved_addition','rewrite_revision1','traversal','self_reference','invalid_hash'])
def test_revision2_rejects_scope_expansion_or_unbound_history(case):
    prior,revision=revision2_fixture();revision=deepcopy(revision)
    if case=='baseline':revision['baseline_commit']=c.ORIGINAL_UNLOCK
    elif case=='prior_commit':revision['prior_implementation_commit']=c.ORIGINAL_IMPL
    elif case=='original_not_prior_hash':revision['modified_files_sha256']['trend_hold_venue_study/data.py']['before']='a'*64
    elif case=='modified_scope':revision['modified_files_sha256'][c.LOCK]={'before':'b'*64,'after':'e'*64}
    elif case=='unapproved_addition':revision['added_files_sha256']['trend_hold_venue_study/alternate_engine.py']='a'*64
    elif case=='rewrite_revision1':revision['added_files_sha256'][c.REVISION+'/notes.md']='a'*64
    elif case=='traversal':revision['added_files_sha256'][c.REVISION2+'/../notes.md']='a'*64
    elif case=='self_reference':revision['added_files_sha256'][c.REVISION2_MANIFEST]='a'*64
    else:revision['modified_files_sha256']['trend_hold_venue_study/data.py']['after']='z'*64
    with pytest.raises(ValueError):c.validate_revision2(revision,prior)


def _write(root,name,data):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)


def _json_bytes(value):
    return (json.dumps(value,sort_keys=True)+'\n').encode()


def registered_tree(tmp_path,monkeypatch):
    original_values={p:('original '+p).encode() for p in c.APPROVED_MODIFIED}
    prior_values={p:('first revision '+p).encode() for p in c.APPROVED_MODIFIED}
    latest_values={p:('second revision '+p).encode() for p in c.APPROVED_MODIFIED}
    original={'baseline_commit':c.CP1,
        'sha256':{p:sha256(v).hexdigest() for p,v in original_values.items()}}
    original['sha256']['immutable.py']=sha256(b'fixed').hexdigest()
    first={'baseline_commit':c.ORIGINAL_UNLOCK,'original_implementation_commit':c.ORIGINAL_IMPL,
        'modified_files_sha256':{p:{'before':original['sha256'][p],'after':sha256(v).hexdigest()}
                                 for p,v in prior_values.items()},
        'added_files_sha256':{c.REVISION+'/notes.md':sha256(b'first notes').hexdigest()}}
    second={'baseline_commit':c.PRIOR_UNLOCK,'prior_implementation_commit':c.PRIOR_IMPL,
        'modified_files_sha256':{p:{'before':sha256(prior_values[p]).hexdigest(),'after':sha256(v).hexdigest()}
                                 for p,v in latest_values.items()},
        'added_files_sha256':{c.REVISION2+'/notes.md':sha256(b'second notes').hexdigest()}}
    original_raw,first_raw,second_raw=map(_json_bytes,(original,first,second))
    for p,v in latest_values.items():_write(tmp_path,p,v)
    for p,v in [('immutable.py',b'fixed'),(c.REVISION+'/notes.md',b'first notes'),
                (c.REVISION2+'/notes.md',b'second notes'),(c.CP2_MANIFEST,original_raw),
                (c.REVISION_MANIFEST,first_raw),(c.REVISION2_MANIFEST,second_raw)]:_write(tmp_path,p,v)
    historical={c.ORIGINAL_IMPL+':'+c.CP2_MANIFEST:original_raw,
                c.PRIOR_IMPL+':'+c.REVISION_MANIFEST:first_raw}
    def fake_git(root,*args):
        assert args[0]=='show'
        return historical[args[1]]
    monkeypatch.setattr(c,'_git',fake_git)
    return original,first,second


def test_verify_cp2_preserves_both_historical_manifests_and_returns_revision2(tmp_path,monkeypatch):
    original,first,second=registered_tree(tmp_path,monkeypatch)
    result=c.verify_cp2(tmp_path)
    assert result['revision']==first
    assert result['revision2']==second
    assert result['sha256'][c.CP2_MANIFEST]==sha256((tmp_path/c.CP2_MANIFEST).read_bytes()).hexdigest()
    assert result['sha256'][c.REVISION_MANIFEST]==sha256((tmp_path/c.REVISION_MANIFEST).read_bytes()).hexdigest()
    assert result['sha256']['immutable.py']==original['sha256']['immutable.py']
    assert result['revision2_manifest_sha256']==sha256((tmp_path/c.REVISION2_MANIFEST).read_bytes()).hexdigest()


@pytest.mark.parametrize('path',[c.CP2_MANIFEST,c.REVISION_MANIFEST])
def test_verify_cp2_rejects_even_whitespace_rewrite_of_historical_manifests(tmp_path,monkeypatch,path):
    registered_tree(tmp_path,monkeypatch)
    with (tmp_path/path).open('ab') as stream:stream.write(b' ')
    with pytest.raises(ValueError,match='manifest changed'):c.verify_cp2(tmp_path)


def _chain(tmp_path,monkeypatch):
    implementation,head='e'*40,'f'*40
    first={'added_files_sha256':{c.REVISION+'/notes.md':'a'*64}}
    second={'added_files_sha256':{c.REVISION2+'/notes.md':'b'*64}}
    raw=_json_bytes(second);_write(tmp_path,c.REVISION2_MANIFEST,raw)
    before=c.FALSE+c.POST
    after=c.TRUE+c.POST
    parents={c.ORIGINAL_IMPL:c.CP1,c.ORIGINAL_UNLOCK:c.ORIGINAL_IMPL,c.PRIOR_IMPL:c.ORIGINAL_UNLOCK,
             c.PRIOR_UNLOCK:c.PRIOR_IMPL,implementation:c.PRIOR_UNLOCK,head:implementation}
    modifications=sorted(c.APPROVED_MODIFIED|{c.LOCK})
    def delta(manifest,added):
        names=['M\t'+p for p in modifications]+['A\t'+p for p in sorted(set(added)|{manifest})]
        return ('\n'.join(names)+'\n').encode()
    table={('status','--porcelain','--untracked-files=all'):b'',
           ('rev-list','--parents','-n','1','HEAD'):(head+' '+implementation+'\n').encode(),
           ('show',implementation+':'+c.REVISION2_MANIFEST):raw}
    for child,parent in parents.items():
        table[('rev-list','--parents','-n','1',child)]=(child+' '+parent+'\n').encode()
    for commit in (c.ORIGINAL_IMPL,c.PRIOR_IMPL,implementation):table[('show',commit+':'+c.LOCK)]=before
    for commit in (c.ORIGINAL_UNLOCK,c.PRIOR_UNLOCK,head):table[('show',commit+':'+c.LOCK)]=after
    for left,right in ((c.ORIGINAL_IMPL,c.ORIGINAL_UNLOCK),(c.PRIOR_IMPL,c.PRIOR_UNLOCK),(implementation,head)):
        table[('diff','--name-status',left,right)]=('M\t'+c.LOCK+'\n').encode()
        table[('diff','--summary',left,right)]=b''
    for left,right,manifest,doc in ((c.ORIGINAL_UNLOCK,c.PRIOR_IMPL,c.REVISION_MANIFEST,first),
                                   (c.PRIOR_UNLOCK,implementation,c.REVISION2_MANIFEST,second)):
        table[('diff','--name-status',left,right)]=delta(manifest,doc['added_files_sha256'])
        table[('diff','--summary',left,right)]=('\n'.join(' create mode 100644 '+p for p in
            sorted(set(doc['added_files_sha256'])|{manifest}))+'\n').encode()
    monkeypatch.setattr(scope,'BTC_IS_UNLOCKED',True)
    monkeypatch.setattr(scope,'POST_IS_AND_ETH_UNLOCKED',False)
    monkeypatch.setattr(c,'_git',lambda root,*args:table[args])
    monkeypatch.setattr(c,'verify_frozen',lambda root,btc_unlocked:dict(locks={'BTC_IS_UNLOCKED':btc_unlocked,'POST_IS_AND_ETH_UNLOCKED':False}))
    monkeypatch.setattr(c,'verify_cp2',lambda root:dict(sha256={'fixture.py':'c'*64},revision=first,revision2=second))
    return table,implementation,head


def test_second_unlock_accepts_only_the_preserved_ancestor_chain(tmp_path,monkeypatch):
    table,implementation,head=_chain(tmp_path,monkeypatch)
    result=c.verify_unlock(tmp_path)
    assert result['implementation_commit']==implementation
    assert result['unlock_commit']==head
    assert result['prior_implementation_commit']==c.PRIOR_IMPL
    assert result['prior_unlock_commit']==c.PRIOR_UNLOCK
    assert result['fixed_account_count']==42


@pytest.mark.parametrize('case',['dirty','parent','old_parent','extra_modification','mode','new_unlock_extra',
                                'historical_post_unlock','revision_manifest','existing_output'])
def test_second_unlock_rejects_chain_scope_or_output_changes(tmp_path,monkeypatch,case):
    table,implementation,head=_chain(tmp_path,monkeypatch)
    if case=='dirty':table[('status','--porcelain','--untracked-files=all')]=b'?? unexpected.txt\n'
    elif case=='parent':table[('rev-list','--parents','-n','1',implementation)]=(implementation+' '+c.ORIGINAL_UNLOCK).encode()
    elif case=='old_parent':table[('rev-list','--parents','-n','1',c.PRIOR_IMPL)]=(c.PRIOR_IMPL+' '+c.CP1).encode()
    elif case=='extra_modification':table[('diff','--name-status',c.PRIOR_UNLOCK,implementation)]+=b'M\tbacktester.py\n'
    elif case=='mode':table[('diff','--summary',c.PRIOR_UNLOCK,implementation)]+=b' mode change 100644 => 100755 trend_hold_venue_study/data.py\n'
    elif case=='new_unlock_extra':table[('diff','--name-status',implementation,head)]+=b'M\ttrend_hold_venue_study/data.py\n'
    elif case=='historical_post_unlock':table[('show',c.ORIGINAL_UNLOCK+':'+c.LOCK)]=c.TRUE+b'POST_IS_AND_ETH_UNLOCKED = True\n'
    elif case=='revision_manifest':(tmp_path/c.REVISION2_MANIFEST).write_bytes(b'{}\n')
    else:(tmp_path/c.OUT).mkdir(parents=True)
    with pytest.raises((ValueError,FileExistsError)):c.verify_unlock(tmp_path)

"""Source-only protection of CP2 history and two approved input revisions."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from . import scope
from trend_hold_study.data import verify_environment

CP1='deadf01a4570386e14bbbec41d6e8eadd3345cac'
ROOT=Path(__file__).resolve().parents[1]
REG='reports/trend_hold_venue_round/cp2_registration'
OUT='reports/trend_hold_venue_round/cp2_btc_is'
LOCK='trend_hold_venue_study/scope.py'
FALSE=b'BTC_IS_UNLOCKED = False\n'
TRUE=b'BTC_IS_UNLOCKED = True\n'
POST=b'POST_IS_AND_ETH_UNLOCKED = False\n'
CP1_MANIFEST='reports/trend_hold_venue_round/implementation_hashes.json'
CP2_MANIFEST=REG+'/implementation_hashes.json'
ORIGINAL_IMPL='4870310dfa9dd4a51b64aaa149caffaf48de79f9'
ORIGINAL_UNLOCK='51a2bc3e15c4a12a87424dc7217278b965c52d81'
REVISION=REG+'/input_revision_1'
REVISION_MANIFEST=REVISION+'/implementation_hashes.json'
APPROVED_MODIFIED={'trend_hold_venue_study/data.py','trend_hold_venue_study/cp2_control.py'}
PRIOR_IMPL='c4c619160bc4c66d643f6bb664b429ed239f6e20'
PRIOR_UNLOCK='ef0d3effc947467fcb91f7cae5a052634a47ae35'
REVISION2=REG+'/input_revision_2'
REVISION2_MANIFEST=REVISION2+'/implementation_hashes.json'
REVISION2_ADDED_SOURCE={
    'trend_hold_venue_study/input_binding.py',
    'tests/test_trend_venue_cp2_daily_binding.py',
    'tests/test_trend_venue_cp2_revision2_control.py',
}

def _git(root,*args):return subprocess.check_output(['git',*args],cwd=root)
def digest(raw):return sha256(raw).hexdigest()

def require_btc_is():
    if not scope.BTC_IS_UNLOCKED or scope.POST_IS_AND_ETH_UNLOCKED:
        raise PermissionError('CP2 requires BTC_IS_UNLOCKED=True AND POST_IS_AND_ETH_UNLOCKED=False')

def validate_transition(before,after):
    if before.splitlines(keepends=True).count(FALSE)!=1 or before.splitlines(keepends=True).count(POST)!=1:
        raise ValueError('implementation must retain both exact False locks')
    if after!=before.replace(FALSE,TRUE):raise ValueError('only the BTC IS single-line unlock is allowed')

def verify_frozen(root=ROOT,*,btc_unlocked=False):
    root=Path(root)
    raw_manifest=(root/CP1_MANIFEST).read_bytes()
    if raw_manifest!=_git(root,'show',CP1+':'+CP1_MANIFEST):raise ValueError('CP1 manifest bytes changed')
    cp1=json.loads(raw_manifest)
    historical=json.loads((root/'reports/trend_hold_venue_round/registration.json').read_text())
    lock=(root/LOCK).read_bytes()
    if lock.splitlines(keepends=True).count(POST)!=1:raise ValueError('POST-IS/ETH lock is not exactly False')
    required=TRUE if btc_unlocked else FALSE
    if lock.splitlines(keepends=True).count(required)!=1:raise ValueError('BTC lock does not match required stage')
    for path,expected in historical['protected_sources_sha256'].items():
        if digest((root/path).read_bytes())!=expected:raise ValueError('historical frozen mismatch: '+path)
    for path,expected in cp1['sha256'].items():
        data=(root/path).read_bytes()
        if path==LOCK and btc_unlocked:data=data.replace(TRUE,FALSE)
        if digest(data)!=expected:raise ValueError('CP1 frozen mismatch: '+path)
    return dict(cp1_commit=CP1,cp1_files=len(cp1['sha256'])+1,historical_files=len(historical['protected_sources_sha256']),
                locks={'BTC_IS_UNLOCKED':btc_unlocked,'POST_IS_AND_ETH_UNLOCKED':False},environment=verify_environment())

def validate_revision(revision,original):
    if (revision.get('baseline_commit')!=ORIGINAL_UNLOCK or
        revision.get('original_implementation_commit')!=ORIGINAL_IMPL):
        raise ValueError('input revision must preserve the original CP2 commits')
    modified=revision.get('modified_files_sha256',{})
    if set(modified)!=APPROVED_MODIFIED:
        raise ValueError('revision may modify only the approved loader and registration controller')
    for path,hashes in modified.items():
        if hashes.get('before')!=original['sha256'].get(path):
            raise ValueError('revision does not bind original registered bytes: '+path)
        if not isinstance(hashes.get('after'),str) or len(hashes['after'])!=64:
            raise ValueError('revision SHA256 required')
    added=revision.get('added_files_sha256',{})
    if not added or REVISION_MANIFEST in added:
        raise ValueError('revision additions must exclude the self-referential manifest')
    for path,value in added.items():
        if not (path.startswith(REVISION+'/') or path in (
            'tests/test_trend_venue_cp2_input_amendment.py',
            'tests/test_trend_venue_cp2_revision_control.py')):
            raise ValueError('unapproved revision addition: '+path)
        if '..' in Path(path).parts or not isinstance(value,str) or len(value)!=64:
            raise ValueError('invalid revision path/hash')
    effective=dict(original['sha256'])
    effective.update({path:hashes['after'] for path,hashes in modified.items()})
    effective.update(added)
    return effective

def _is_sha256(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def validate_revision2(revision,prior_effective):
    """Bind approved new bytes to the first revision, preserving earlier files."""
    if (revision.get('baseline_commit')!=PRIOR_UNLOCK or
        revision.get('prior_implementation_commit')!=PRIOR_IMPL):
        raise ValueError('second input revision must preserve the first revision commits')
    modified=revision.get('modified_files_sha256',{})
    if set(modified)!=APPROVED_MODIFIED:
        raise ValueError('second revision may modify only the approved loader and registration controller')
    for path,hashes in modified.items():
        if hashes.get('before')!=prior_effective.get(path):
            raise ValueError('second revision does not bind first-revision registered bytes: '+path)
        if not _is_sha256(hashes.get('after')):raise ValueError('second revision SHA256 required')
    added=revision.get('added_files_sha256',{})
    if not added or REVISION2_MANIFEST in added:
        raise ValueError('second revision additions must exclude the self-referential manifest')
    for path,value in added.items():
        if not (path.startswith(REVISION2+'/') or path in REVISION2_ADDED_SOURCE):
            raise ValueError('unapproved second revision addition: '+path)
        if '..' in Path(path).parts or not _is_sha256(value) or path in prior_effective:
            raise ValueError('invalid or previously registered second revision path/hash')
    effective=dict(prior_effective)
    effective.update({path:hashes['after'] for path,hashes in modified.items()})
    effective.update(added)
    return effective

def verify_cp2(root=ROOT):
    root=Path(root);raw=(root/CP2_MANIFEST).read_bytes();doc=json.loads(raw)
    if doc['baseline_commit']!=CP1:raise ValueError('CP2 baseline mismatch')
    if raw!=_git(root,'show',ORIGINAL_IMPL+':'+CP2_MANIFEST):
        raise ValueError('original CP2 registration manifest changed')
    revision_raw=(root/REVISION_MANIFEST).read_bytes()
    if revision_raw!=_git(root,'show',PRIOR_IMPL+':'+REVISION_MANIFEST):
        raise ValueError('first input revision manifest changed')
    revision=json.loads(revision_raw)
    prior=validate_revision(revision,doc)
    prior[CP2_MANIFEST]=digest(raw)
    prior[REVISION_MANIFEST]=digest(revision_raw)
    revision2_raw=(root/REVISION2_MANIFEST).read_bytes()
    revision2=json.loads(revision2_raw)
    expected=validate_revision2(revision2,prior)
    for path,want in expected.items():
        if digest((root/path).read_bytes())!=want:raise ValueError('CP2 implementation mismatch: '+path)
    return {**doc,'sha256':expected,'revision':revision,'revision_manifest_sha256':digest(revision_raw),
            'revision2':revision2,'revision2_manifest_sha256':digest(revision2_raw)}

def _verify_unlock_pair(root,implementation,unlocked):
    if _git(root,'diff','--name-status',implementation,unlocked).decode().strip()!='M\t'+LOCK:
        raise ValueError('unlock must change only the BTC lock file')
    if _git(root,'diff','--summary',implementation,unlocked).strip():
        raise ValueError('unlock cannot change file mode')
    validate_transition(_git(root,'show',implementation+':'+LOCK),_git(root,'show',unlocked+':'+LOCK))

def _verify_revision_delta(root,baseline,implementation,revision,manifest):
    changes=_git(root,'diff','--name-status',baseline,implementation).decode().splitlines()
    expected_mod=APPROVED_MODIFIED|{LOCK}
    expected_add=set(revision['added_files_sha256'])|{manifest}
    actual_mod={line[2:] for line in changes if line.startswith('M\t')}
    actual_add={line[2:] for line in changes if line.startswith('A\t')}
    if actual_mod!=expected_mod or actual_add!=expected_add or len(changes)!=len(expected_mod)+len(expected_add):
        raise ValueError('unregistered or unauthorized revision changes')
    summary=_git(root,'diff','--summary',baseline,implementation).decode().splitlines()
    expected_summary={'create mode 100644 '+path for path in expected_add}
    if {line.strip() for line in summary}!=expected_summary or len(summary)!=len(expected_summary):
        raise ValueError('revision file mode changes forbidden')

def verify_unlock(root=ROOT):
    require_btc_is()
    root=Path(root)
    if _git(root,'status','--porcelain','--untracked-files=all').strip():raise ValueError('formal run requires clean worktree')
    head_parts=_git(root,'rev-list','--parents','-n','1','HEAD').decode().split()
    if len(head_parts)!=2:raise ValueError('unlock must have one implementation parent')
    head,implementation=head_parts
    for commit,parent in ((ORIGINAL_IMPL,CP1),(ORIGINAL_UNLOCK,ORIGINAL_IMPL),
                          (PRIOR_IMPL,ORIGINAL_UNLOCK),(PRIOR_UNLOCK,PRIOR_IMPL),
                          (implementation,PRIOR_UNLOCK)):
        if _git(root,'rev-list','--parents','-n','1',commit).decode().split()!=[commit,parent]:
            raise ValueError('approved CP2 ancestor chain changed: '+commit)
    for before,after in ((ORIGINAL_IMPL,ORIGINAL_UNLOCK),(PRIOR_IMPL,PRIOR_UNLOCK),(implementation,head)):
        _verify_unlock_pair(root,before,after)
    # Every re-lock restores the same scope bytes; POST-IS/ETH never unlocks.
    validate_transition(_git(root,'show',PRIOR_IMPL+':'+LOCK),_git(root,'show',ORIGINAL_UNLOCK+':'+LOCK))
    validate_transition(_git(root,'show',implementation+':'+LOCK),_git(root,'show',PRIOR_UNLOCK+':'+LOCK))
    if (root/REVISION2_MANIFEST).read_bytes()!=_git(root,'show',implementation+':'+REVISION2_MANIFEST):
        raise ValueError('second revision manifest differs from implementation commit')
    protected=verify_frozen(root,btc_unlocked=True);doc=verify_cp2(root)
    _verify_revision_delta(root,ORIGINAL_UNLOCK,PRIOR_IMPL,doc['revision'],REVISION_MANIFEST)
    _verify_revision_delta(root,PRIOR_UNLOCK,implementation,doc['revision2'],REVISION2_MANIFEST)
    if os.path.lexists(root/OUT):raise FileExistsError('formal output exists; no automatic resume')
    return {**protected,'original_implementation_commit':ORIGINAL_IMPL,'original_unlock_commit':ORIGINAL_UNLOCK,
            'prior_implementation_commit':PRIOR_IMPL,'prior_unlock_commit':PRIOR_UNLOCK,
            'implementation_commit':implementation,'unlock_commit':head,'starting_worktree_status':'',
            'cp2_files':len(doc['sha256'])+1,'fixed_account_count':42,
            'approved_input_revision':'eight exact Spot close timestamps; one exact pre-Context CM funding record',
            'approved_input_revision2':'four official CM daily archives; UM canonical-content and Parquet-byte hash binding'}

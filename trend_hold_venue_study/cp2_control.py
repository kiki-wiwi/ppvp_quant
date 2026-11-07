"""Source-only protection of original CP2 history and the approved input revision."""
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

def verify_cp2(root=ROOT):
    root=Path(root);raw=(root/CP2_MANIFEST).read_bytes();doc=json.loads(raw)
    if doc['baseline_commit']!=CP1:raise ValueError('CP2 baseline mismatch')
    if raw!=_git(root,'show',ORIGINAL_IMPL+':'+CP2_MANIFEST):
        raise ValueError('original CP2 registration manifest changed')
    revision_raw=(root/REVISION_MANIFEST).read_bytes()
    revision=json.loads(revision_raw)
    expected=validate_revision(revision,doc)
    for path,want in expected.items():
        if digest((root/path).read_bytes())!=want:raise ValueError('CP2 implementation mismatch: '+path)
    expected[CP2_MANIFEST]=digest(raw)
    return {**doc,'sha256':expected,'revision':revision,'revision_manifest_sha256':digest(revision_raw)}

def verify_unlock(root=ROOT):
    require_btc_is()
    root=Path(root)
    if _git(root,'status','--porcelain','--untracked-files=all').strip():raise ValueError('formal run requires clean worktree')
    head_parts=_git(root,'rev-list','--parents','-n','1','HEAD').decode().split()
    if len(head_parts)!=2:raise ValueError('unlock must have one implementation parent')
    head,implementation=head_parts
    if _git(root,'rev-list','--parents','-n','1',ORIGINAL_IMPL).decode().split()!=[ORIGINAL_IMPL,CP1]:
        raise ValueError('original implementation history changed')
    if _git(root,'rev-list','--parents','-n','1',ORIGINAL_UNLOCK).decode().split()!=[ORIGINAL_UNLOCK,ORIGINAL_IMPL]:
        raise ValueError('original unlock history changed')
    if _git(root,'rev-list','--parents','-n','1',implementation).decode().split()!=[implementation,ORIGINAL_UNLOCK]:
        raise ValueError('approved revision must directly descend from original unlock')
    if _git(root,'diff','--name-status',implementation,head).decode().strip()!='M\t'+LOCK:
        raise ValueError('unlock must change only the BTC lock file')
    if _git(root,'diff','--summary',implementation,head).strip():raise ValueError('unlock cannot change file mode')
    validate_transition(_git(root,'show',implementation+':'+LOCK),_git(root,'show',head+':'+LOCK))
    validate_transition(_git(root,'show',implementation+':'+LOCK),_git(root,'show',ORIGINAL_UNLOCK+':'+LOCK))
    if (root/REVISION_MANIFEST).read_bytes()!=_git(root,'show',implementation+':'+REVISION_MANIFEST):
        raise ValueError('revision manifest differs from implementation commit')
    protected=verify_frozen(root,btc_unlocked=True);doc=verify_cp2(root)
    revision=doc['revision']
    changes=_git(root,'diff','--name-status',ORIGINAL_UNLOCK,implementation).decode().splitlines()
    expected_mod=APPROVED_MODIFIED|{LOCK}
    expected_add=set(revision['added_files_sha256'])|{REVISION_MANIFEST}
    actual_mod={line[2:] for line in changes if line.startswith('M\t')}
    actual_add={line[2:] for line in changes if line.startswith('A\t')}
    if actual_mod!=expected_mod or actual_add!=expected_add or len(changes)!=len(expected_mod)+len(expected_add):
        raise ValueError('unregistered or unauthorized revision changes')
    summary=_git(root,'diff','--summary',ORIGINAL_UNLOCK,implementation).decode().splitlines()
    if any('create mode 100644 ' not in line for line in summary):raise ValueError('revision file mode changes forbidden')
    if os.path.lexists(root/OUT):raise FileExistsError('formal output exists; no automatic resume')
    return {**protected,'original_implementation_commit':ORIGINAL_IMPL,'original_unlock_commit':ORIGINAL_UNLOCK,
            'implementation_commit':implementation,'unlock_commit':head,'starting_worktree_status':'',
            'cp2_files':len(doc['sha256'])+1,'fixed_account_count':42,
            'approved_input_revision':'eight exact Spot close timestamps; one exact pre-Context CM funding record'}

"""Source-only protection of the fixed CP1 -> CP2 implementation -> unlock chain."""
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

def verify_cp2(root=ROOT):
    root=Path(root);doc=json.loads((root/CP2_MANIFEST).read_text())
    if doc['baseline_commit']!=CP1:raise ValueError('CP2 baseline mismatch')
    for path,expected in doc['sha256'].items():
        if digest((root/path).read_bytes())!=expected:raise ValueError('CP2 implementation mismatch: '+path)
    return doc

def verify_unlock(root=ROOT):
    require_btc_is()
    root=Path(root)
    if _git(root,'status','--porcelain','--untracked-files=all').strip():raise ValueError('formal run requires clean worktree')
    head_parts=_git(root,'rev-list','--parents','-n','1','HEAD').decode().split()
    if len(head_parts)!=2:raise ValueError('unlock must have one implementation parent')
    head,implementation=head_parts
    if _git(root,'rev-list','--parents','-n','1',implementation).decode().split()!=[implementation,CP1]:
        raise ValueError('CP2 implementation must directly descend from CP1')
    changes=_git(root,'diff','--name-status',CP1,implementation).decode().splitlines()
    if not changes or any(not line.startswith('A\t') for line in changes):raise ValueError('CP2 implementation must be additions only')
    if _git(root,'diff','--name-status',implementation,head).decode().strip()!='M\t'+LOCK:
        raise ValueError('unlock must change only the BTC lock file')
    if _git(root,'diff','--summary',implementation,head).strip():raise ValueError('unlock cannot change file mode')
    validate_transition(_git(root,'show',implementation+':'+LOCK),_git(root,'show',head+':'+LOCK))
    if (root/CP2_MANIFEST).read_bytes()!=_git(root,'show',implementation+':'+CP2_MANIFEST):
        raise ValueError('CP2 manifest differs from implementation commit')
    protected=verify_frozen(root,btc_unlocked=True);doc=verify_cp2(root)
    expected=set(doc['sha256'])|{CP2_MANIFEST}
    if {line.split('\t',1)[1] for line in changes}!=expected:raise ValueError('unregistered implementation additions')
    if os.path.lexists(root/OUT):raise FileExistsError('formal output exists; no automatic resume')
    return {**protected,'implementation_commit':implementation,'unlock_commit':head,'starting_worktree_status':'',
            'cp2_files':len(expected),'fixed_account_count':42}

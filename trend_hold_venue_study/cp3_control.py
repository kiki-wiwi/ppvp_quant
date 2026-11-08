"""CP3 source/registration protection and exact one-line authorization protocol."""
from hashlib import sha256
from pathlib import Path
from datetime import datetime,timezone
import importlib.abc
import json
import subprocess
import sys
from . import scope, cp2_control
from trend_hold_study.data import verify_environment

BASELINE='f0c21acdd9320d87272d3c8cd46fe95b8d72ec5d'
ROOT=Path(__file__).resolve().parents[1]
REG='reports/trend_hold_venue_round/cp3_registration'
OUT='reports/trend_hold_venue_round/cp3_post_is_and_eth'
LOCK='trend_hold_venue_study/scope.py'
MANIFEST=REG+'/implementation_hashes.json'
INPUT_MANIFEST=REG+'/venue_cp3_inputs.json'
CP2_DELIVERY='reports/trend_hold_venue_round/cp2_btc_is/delivery_hashes.json'
CP2_DELIVERY_SHA256='c6f1762f3add25eae7788c997c84bdb7bba00080ddc9b40b1a0a88c34f540084'
BTC_FALSE=b'BTC_IS_UNLOCKED = False\n'
BTC_TRUE=b'BTC_IS_UNLOCKED = True\n'
POST_FALSE=b'POST_IS_AND_ETH_UNLOCKED = False\n'
POST_TRUE=b'POST_IS_AND_ETH_UNLOCKED = True\n'
EXCLUSIONS={
 'tests/test_trend_hold_engine.py::test_rejects_missing_or_invalid_daily_inputs_and_real_context_before_access':'Historical R5 unlocked baseline conflicts with the old pre-unlock assertion.',
 'tests/test_trend_venue_scope_features.py::test_two_real_locks_and_provenance':'Historical CP1 assertion requires BTC_IS_UNLOCKED=False; approved current baseline is True.',
 'tests/test_trend_venue_cp2_runtime.py::test_runtime_rejects_post_is_eth_and_real_locked_before_feature_work':'Historical CP2 pre-unlock entry assertion conflicts with BTC_IS_UNLOCKED=True.'}
BLOCKED_PREPARATION_MODULES=('trend_hold_venue_study.features','trend_hold_venue_study.linear',
 'trend_hold_venue_study.portfolios','trend_hold_venue_study.runtime','trend_hold_venue_study.statistics',
 'trend_hold_venue_study.cp3_runtime','trend_hold_venue_study.cp3_analysis','trend_hold_venue_study.cp3_plotting',
 'trend_hold_study.features','trend_hold_study.engine','trend_hold_study.statistics','impulse_study.features','impulse_study.engine')


def require_preparation():
    if not scope.BTC_IS_UNLOCKED or scope.POST_IS_AND_ETH_UNLOCKED:
        raise scope.RealDataLockedError('CP3 preparation requires BTC=True and POST=False')


def require_cp3():
    if not scope.BTC_IS_UNLOCKED or not scope.POST_IS_AND_ETH_UNLOCKED:
        raise scope.RealDataLockedError('CP3 real inputs require both independent authorizations True')


def _git(root,*args):return subprocess.check_output(['git',*args],cwd=root)


def digest(path):
    h=sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def normalize_scope(raw,*,post_unlocked):
    lines=raw.splitlines(keepends=True)
    expected=POST_TRUE if post_unlocked else POST_FALSE
    if lines.count(BTC_TRUE)!=1 or lines.count(expected)!=1:
        raise ValueError('physical lock lines differ from required CP3 stage')
    if BTC_FALSE in lines or (POST_FALSE if post_unlocked else POST_TRUE) in lines:
        raise ValueError('duplicate/inconsistent lock declaration')
    return b''.join(BTC_FALSE if line==BTC_TRUE else POST_FALSE if line==POST_TRUE else line for line in lines)


def validate_transition(before,after):
    normalize_scope(before,post_unlocked=False)
    normalize_scope(after,post_unlocked=True)
    if after!=before.replace(POST_FALSE,POST_TRUE):raise ValueError('only the exact POST lock line may change')


def allowed_addition(path):
    parts=Path(path).parts
    if '..' in parts or Path(path).is_absolute():return False
    return (path.startswith(REG+'/') or
        path.startswith('trend_hold_venue_study/cp3_') and path.endswith('.py') or
        path.startswith('tests/test_trend_venue_cp3_') and path.endswith('.py') or
        path=='tools/run_trend_hold_venue_cp3.py')


def verify_frozen(root=ROOT,*,post_unlocked=False,verify_deliveries=True):
    root=Path(root)
    cp1_path=cp2_control.CP1_MANIFEST
    raw=(root/cp1_path).read_bytes()
    if raw!=_git(root,'show',cp2_control.CP1+':'+cp1_path):raise ValueError('CP1 implementation manifest changed')
    historical=json.loads((root/'reports/trend_hold_venue_round/registration.json').read_text())
    for path,expected in historical['protected_sources_sha256'].items():
        if digest(root/path)!=expected:raise ValueError('historical source changed: '+path)
    lock_normal=normalize_scope((root/LOCK).read_bytes(),post_unlocked=post_unlocked)
    cp1=json.loads(raw)
    for path,expected in cp1['sha256'].items():
        actual=sha256(lock_normal).hexdigest() if path==LOCK else digest(root/path)
        if actual!=expected:raise ValueError('CP1 frozen bytes changed: '+path)
    cp2=cp2_control.verify_cp2(root)
    revision2=cp2_control.REVISION2_MANIFEST
    if (root/revision2).read_bytes()!=_git(root,'show',BASELINE+':'+revision2):
        raise ValueError('final CP2 revision manifest changed')
    for row in _git(root,'diff','--name-status','--no-renames',BASELINE,'--').decode().splitlines():
        status,path=row.split('\t',1)
        if path==LOCK and status=='M' and post_unlocked:continue
        if status=='A' and allowed_addition(path):continue
        raise ValueError('change outside new CP3 implementation scope: '+row)
    if digest(root/CP2_DELIVERY)!=CP2_DELIVERY_SHA256:raise ValueError('CP2 delivery manifest changed')
    delivery=json.loads((root/CP2_DELIVERY).read_text())
    if verify_deliveries:
        for number,(path,expected) in enumerate(delivery['sha256'].items(),1):
            file=root/path
            if file.stat().st_size!=delivery['size_bytes'][path] or digest(file)!=expected:
                raise ValueError('CP2 output changed: '+path)
            if number%100==0:print('verified frozen CP2 delivery files',number,flush=True)
    return {'baseline_commit':BASELINE,'cp1_files':len(cp1['sha256'])+1,
        'historical_files':len(historical['protected_sources_sha256']),
        'cp2_registration_files':len(cp2['sha256'])+1,
        'cp2_delivery_files':len(delivery['sha256']),'cp2_deliveries_verified':verify_deliveries,
        'cp2_delivery_manifest_sha256':CP2_DELIVERY_SHA256,'environment':verify_environment(),
        'locks':{'BTC_IS_UNLOCKED':True,'POST_IS_AND_ETH_UNLOCKED':post_unlocked}}


class PreparationImportBarrier(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in BLOCKED_PREPARATION_MODULES:
            raise PermissionError('strategy import forbidden during locked input preparation: '+fullname)


def install_preparation_barrier():
    loaded=sorted(set(sys.modules).intersection(BLOCKED_PREPARATION_MODULES))
    if loaded:raise PermissionError('strategy modules already loaded before locked preparation: '+','.join(loaded))
    barrier=PreparationImportBarrier();sys.meta_path.insert(0,barrier)
    return barrier


def register_implementation(root=ROOT):
    require_preparation();root=Path(root)
    if _git(root,'rev-parse','HEAD').decode().strip()!=BASELINE:raise ValueError('implementation must start from approved CP2 baseline')
    target=root/MANIFEST
    if target.exists():raise FileExistsError('CP3 implementation already registered; no refreezing')
    validation=json.loads((root/REG/'synthetic_validation.json').read_text())
    if validation.get('status')!='passed' or set(validation.get('deselected_nodes',[]))!=set(EXCLUSIONS):
        raise ValueError('required synthetic validation and exact three exclusions must pass first')
    if not (root/INPUT_MANIFEST).is_file():raise ValueError('completed input registration required')
    frozen=verify_frozen(root,post_unlocked=False)
    paths=_git(root,'ls-files','--others','--exclude-standard').decode().splitlines()
    if not paths or any(not allowed_addition(p) for p in paths):raise ValueError('unregistered file outside implementation scope')
    protected={p:digest(root/p) for p in paths}
    # Every registration evidence file must be tracked, including any ignored log.
    for p in (root/REG).rglob('*'):
        if p.is_file() and str(p.relative_to(root)) not in protected:
            raise ValueError('registration evidence excluded by Git rules: '+str(p.relative_to(root)))
    document={'stage':'CP3','baseline_commit':BASELINE,'created_at':datetime.now(timezone.utc).isoformat(),
        'declaration':'User-authorized one-shot descriptive evaluation; R5 failed IS V3/V5; no automatic promotion.',
        'input_manifest':INPUT_MANIFEST,'input_manifest_sha256':digest(root/INPUT_MANIFEST),
        'frozen_verification':frozen,'deselected_nodes':EXCLUSIONS,'sha256':protected,
        'self_omitted':MANIFEST,'physical_lock_state':{'BTC_IS_UNLOCKED':True,'POST_IS_AND_ETH_UNLOCKED':False}}
    with target.open('x') as stream:json.dump(document,stream,indent=2);stream.write('\n')
    return document


def verify_implementation(root=ROOT):
    root=Path(root);doc=json.loads((root/MANIFEST).read_text())
    if doc['baseline_commit']!=BASELINE or MANIFEST in doc['sha256']:raise ValueError('invalid CP3 registration origin/self reference')
    if not doc['sha256'] or any(not allowed_addition(p) for p in doc['sha256']):raise ValueError('unapproved registered path')
    for path,expected in doc['sha256'].items():
        if digest(root/path)!=expected:raise ValueError('registered CP3 file changed: '+path)
    if digest(root/INPUT_MANIFEST)!=doc['input_manifest_sha256']:raise ValueError('input registration changed')
    return doc


def verify_unlock(root=ROOT):
    require_cp3();root=Path(root)
    if _git(root,'status','--porcelain','--untracked-files=all').strip():raise ValueError('formal run requires a completely clean worktree')
    head=_git(root,'rev-parse','HEAD').decode().strip();implementation=_git(root,'rev-parse','HEAD^').decode().strip()
    if _git(root,'rev-parse',implementation+'^').decode().strip()!=BASELINE:
        raise ValueError('implementation/unlock must be the two direct successors of the approved CP2 baseline')
    if _git(root,'diff','--name-status',implementation,head).decode().strip()!='M\t'+LOCK:
        raise ValueError('unlock commit may change only scope.py')
    if _git(root,'diff','--summary',implementation,head).strip():raise ValueError('unlock cannot change file mode')
    validate_transition(_git(root,'show',implementation+':'+LOCK),_git(root,'show',head+':'+LOCK))
    registered=verify_implementation(root)
    if (root/MANIFEST).read_bytes()!=_git(root,'show',implementation+':'+MANIFEST):raise ValueError('implementation manifest altered')
    changed=_git(root,'diff','--name-status','--no-renames',BASELINE,implementation).decode().splitlines()
    expected=set(registered['sha256'])|{MANIFEST}
    if any(not row.startswith('A\t') for row in changed) or {row.split('\t',1)[1] for row in changed}!=expected:
        raise ValueError('implementation commit additions differ from registered inventory')
    frozen=verify_frozen(root,post_unlocked=True)
    delivery=json.loads((root/CP2_DELIVERY).read_text())
    return {**frozen,'implementation_commit':implementation,'unlock_commit':head,'starting_git_status':'',
        'implementation_manifest_sha256':digest(root/MANIFEST),'cp2_delivery_sha256':delivery['sha256'],
        'cp2_comparison_sha256':delivery['sha256']['reports/trend_hold_venue_round/cp2_btc_is/comparison.csv']}

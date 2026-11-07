"""Source-only registration verification. Never opens market inputs."""
from hashlib import sha256
import json
from pathlib import Path
from .config import BASELINE

REPORT=Path('reports/trend_hold_venue_round')

def digest(path,normalize=False):
    data=Path(path).read_bytes()
    if normalize:
        for name in (b'BTC_IS_UNLOCKED',b'POST_IS_AND_ETH_UNLOCKED'):
            data=data.replace(name+b' = True',name+b' = False')
    return sha256(data).hexdigest()

def verify(root,registration=None,implementation=None):
    root=Path(root)
    registered=registration or json.loads((root/REPORT/'registration.json').read_text())
    if registered['baseline_commit']!=BASELINE: raise ValueError('baseline mismatch')
    for path,expected in registered['protected_sources_sha256'].items():
        if digest(root/path)!=expected: raise ValueError('protected source mismatch: '+path)
    own=implementation or json.loads((root/REPORT/'implementation_hashes.json').read_text())
    for path,expected in own['sha256'].items():
        if digest(root/path,normalize=path=='trend_hold_venue_study/scope.py')!=expected:
            raise ValueError('implementation mismatch: '+path)
    return dict(status='verified',baseline=BASELINE,protected_files=len(registered['protected_sources_sha256']),
        implementation_files=len(own['sha256']))

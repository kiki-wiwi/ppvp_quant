"""Extract selected objects (by sha256) from the audit packs into an output dir."""
import json, sys, tarfile, hashlib
from pathlib import Path
repo = Path(sys.argv[1]); out = Path(sys.argv[2]); wanted = set(json.load(open(sys.argv[3])))
idx = json.loads((repo/'reports/external_audit_20261108/inputs/index.json').read_text())
bypack = {}
for o in idx['objects']:
    if o['sha256'] in wanted: bypack.setdefault(o['pack'], {})[o['member']] = o['sha256']
out.mkdir(parents=True, exist_ok=True); got = 0
for pack, members in bypack.items():
    with tarfile.open(repo/pack, 'r|*') as t:
        for m in t:
            if m.name in members:
                data = t.extractfile(m).read()
                assert hashlib.sha256(data).hexdigest() == members[m.name]
                (out/members[m.name]).write_bytes(data); got += 1
print('wanted', len(wanted), 'extracted', got, 'not in packs', len(wanted - {s for p in bypack.values() for s in p.values()}))

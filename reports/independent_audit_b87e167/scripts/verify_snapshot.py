"""Independent check: snapshot.json vs git tree of anchor commit; LFS pointers; packs."""
import hashlib, json, subprocess, sys, tarfile, re
from pathlib import Path
repo = Path(sys.argv[1]); anchor = sys.argv[2]
snap = json.loads((repo/'reports/external_audit_20261108/snapshot.json').read_text())
idx = json.loads((repo/'reports/external_audit_20261108/inputs/index.json').read_text())
print('snapshot keys', list(snap.keys()), 'files', len(snap['files']))
# 1. tree listing at anchor
tree = subprocess.run(['git','-C',str(repo),'ls-tree','-r','-l',anchor],capture_output=True,text=True,check=True).stdout.splitlines()
tree_paths = {}
for line in tree:
    meta, path = line.split('\t',1)
    mode, typ, sha, size = meta.split()
    tree_paths[path] = (mode, sha, size)
snap_paths = {f['path']: f for f in snap['files']}
print('tree files', len(tree_paths), 'snapshot files', len(snap_paths))
print('in tree not snapshot', sorted(set(tree_paths)-set(snap_paths))[:20])
print('in snapshot not tree', sorted(set(snap_paths)-set(tree_paths))[:20])
# 2. blob content hash vs snapshot (pointer for LFS)
p = subprocess.Popen(['git','-C',str(repo),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
mism=[]; lfs=0; lfs_bad=[]; nonlfs=0
for path,(mode,sha,size) in sorted(tree_paths.items()):
    p.stdin.write((sha+'\n').encode()); p.stdin.flush()
    header = p.stdout.readline().split()
    n = int(header[2]); data = p.stdout.read(n); p.stdout.read(1)
    item = snap_paths.get(path)
    if item is None: continue
    if data.startswith(b'version https://git-lfs.github.com/spec/v1'):
        lfs += 1
        oid = re.search(rb'oid sha256:([0-9a-f]{64})', data).group(1).decode()
        sz = int(re.search(rb'size (\d+)', data).group(1))
        if oid != item['sha256'] or sz != item['size']:
            lfs_bad.append(path)
    else:
        nonlfs += 1
        if hashlib.sha256(data).hexdigest() != item['sha256'] or len(data)!=item['size']:
            mism.append(path)
print('nonlfs checked', nonlfs, 'mismatch', len(mism), mism[:5])
print('lfs pointers', lfs, 'oid/size mismatch', len(lfs_bad), lfs_bad[:5])
print('total snapshot bytes', sum(f['size'] for f in snap['files']))
# 3. packs
objs = {o['member']:(o['sha256'],o['size'],o['pack']) for o in idx['objects']}
seen=0; bad=[]
for pk in idx['packs']:
    with tarfile.open(repo/pk['path'],'r|*') as t:
        for m in t:
            f = t.extractfile(m); h=hashlib.sha256(); n=0
            while (c:=f.read(1<<20)): h.update(c); n+=len(c)
            exp = objs.get(m.name)
            if exp is None or exp[0]!=h.hexdigest() or exp[1]!=n or exp[2]!=pk['path'] or m.name!='objects/'+h.hexdigest():
                bad.append(m.name)
            seen+=1
print('pack members', seen, 'index objects', len(objs), 'bad', len(bad), bad[:5])

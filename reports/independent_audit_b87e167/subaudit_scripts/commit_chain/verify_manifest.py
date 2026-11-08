import sys, re, json, subprocess, hashlib
repo='/home/user/ppvp_quant'
def blob(c,p):
    r=subprocess.run(['git','-C',repo,'cat-file','-p',f'{c}:{p}'],capture_output=True)
    return r.stdout if r.returncode==0 else None
def tree_files(c, d):
    r=subprocess.run(['git','-C',repo,'ls-tree','-r','--name-only',c,'--',d],capture_output=True)
    return set(r.stdout.decode().splitlines())
LFS=re.compile(rb'^version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n?$')
def content_sha(b):
    m=LFS.match(b)
    if m: return m.group(1).decode(), True
    return hashlib.sha256(b).hexdigest(), False
c, man, keypath, base = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
raw=blob(c,man)
print('manifest sha256', hashlib.sha256(raw).hexdigest())
d=json.loads(raw)
for k in keypath.split('.'):
    if k: d=d[k]
items = d.items() if isinstance(d,dict) else [(x['path'],x['sha256']) for x in d]
ok=bad=miss=lfs=0; badl=[]
covered=set()
for p,h in items:
    if isinstance(h,dict): h=h.get('sha256')
    full = (base.rstrip('/')+'/'+p) if base else p
    covered.add(full)
    b=blob(c,full)
    if b is None: miss+=1; badl.append(('MISSING',full)); continue
    s,isl=content_sha(b); lfs+=isl
    if s==h: ok+=1
    else: bad+=1; badl.append(('MISMATCH',full))
print(f'entries={ok+bad+miss} ok={ok} mismatch={bad} missing={miss} lfs_pointer_entries={lfs}')
for x in badl[:20]: print(' ',x)
if base:
    present=tree_files(c, base)
    unc=sorted(present-covered)
    print(f'files under {base} at {c[:7]} not in manifest ({len(unc)}):', unc[:40])

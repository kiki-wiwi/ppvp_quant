import sys, re, json, subprocess, hashlib, itertools
repo='/home/user/ppvp_quant'
def git(*a):
    r=subprocess.run(['git','-C',repo,*a],capture_output=True)
    return r
def blob(c,p):
    r=git('cat-file','-p',f'{c}:{p}')
    return r.stdout if r.returncode==0 else None
def get_map(c, path, keypath):
    d=json.loads(blob(c,path))
    for k in keypath.split('.'):
        if k: d=d[k]
    return d
LOCK_RE=re.compile(rb'^([A-Z_]*UNLOCKED[A-Z_]*\s*(?::\s*[A-Za-z]+\s*)?=\s*)True\b', re.M)
def check(b, h):
    if hashlib.sha256(b).hexdigest()==h: return 'OK'
    locks=list(LOCK_RE.finditer(b))
    for n in range(1,len(locks)+1):
        for sub in itertools.combinations(locks,n):
            bb=bytearray(b)
            for m in sorted(sub,key=lambda m:-m.start()):
                s,e=m.span()
                bb[s:e]=m.group(1)+b'False'
            if hashlib.sha256(bytes(bb)).hexdigest()==h:
                names=[m.group(1).split(b'=')[0].strip().decode() for m in sub]
                return 'OK(norm:'+'+'.join(names)+')'
    return 'MISMATCH'
reg_commit, reg_path, keypath = sys.argv[1], sys.argv[2], sys.argv[3]
commits = sys.argv[4].split(',')
impl = sys.argv[5] if len(sys.argv)>5 else None
m = get_map(reg_commit, reg_path, keypath)
print(f'registry {reg_path}@{reg_commit[:7]} key={keypath}: {len(m)} entries')
summary={}
for p,h in sorted(m.items()):
    if isinstance(h,dict): h=h.get('sha256')
    res=[]
    for c in commits:
        b=blob(c,p)
        r='MISSING' if b is None else check(b,h)
        res.append(f'{c[:7]}={r}')
        summary.setdefault(c[:7],{}).setdefault(r.split('(')[0],0)
        summary[c[:7]][r.split('(')[0]]+=1
    line=' '.join(res)
    if 'MISMATCH' in line or 'MISSING' in line or 'norm' in line: print(' ',p,line)
for c,s in summary.items(): print('  summary',c,s)
if impl:
    r=git('show','--no-renames','--name-status','--format=',impl)
    added=[l.split('\t',1)[1] for l in r.stdout.decode().splitlines() if l.startswith('A')]
    not_cov=[a for a in added if a not in m]
    print(f'  files added in {impl[:7]} not covered by registry ({len(not_cov)}):', not_cov)
    reg_not_added=[p for p in m if p not in added]
    print(f'  registry entries not added in {impl[:7]} (pre-existing deps) ({len(reg_not_added)}):', reg_not_added)

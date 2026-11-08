import sys, re, subprocess, hashlib
repo='/home/user/ppvp_quant'
def blob(c,p):
    r=subprocess.run(['git','-C',repo,'cat-file','-p',f'{c}:{p}'],capture_output=True)
    return r.stdout if r.returncode==0 else None
plan_commit, plan_path = sys.argv[1], sys.argv[2]
commits = sys.argv[3].split(',')
normalize = dict(x.split('=',1) for x in sys.argv[4:]) if len(sys.argv)>4 else {}
txt = blob(plan_commit, plan_path).decode()
rows = re.findall(r'^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|', txt, re.M)
print(f'{len(rows)} rows')
bad=0
for p,h in rows:
    out=[]
    for c in commits:
        b=blob(c,p)
        if b is None: out.append(f'{c[:7]}=MISSING'); bad+=1; continue
        s=hashlib.sha256(b).hexdigest()
        if s!=h and p in normalize:
            old,new=normalize[p].split('|')
            b2=b.replace(new.encode(),old.encode(),1)
            s2=hashlib.sha256(b2).hexdigest()
            if s2==h: out.append(f'{c[:7]}=OK(normalized)'); continue
        if s==h: out.append(f'{c[:7]}=OK')
        else: out.append(f'{c[:7]}=MISMATCH'); bad+=1
    print(p, ' '.join(out))
print('bad', bad)

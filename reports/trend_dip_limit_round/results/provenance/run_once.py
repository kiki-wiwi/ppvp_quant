from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,os,sys,time
ROOT=Path('/home/ubuntu/.codex/worktrees/channel-vah-prereg/ppvp_quant')
AUDIT=Path(__file__).parent
PYTHON='/home/ubuntu/Documents/Codex/2026-11-06/github-ppvp-quant-agent-2-checkpoint/work/diag_env/pandas3.0.5_py314/bin/python'
state={'started_at':datetime.now(timezone.utc).isoformat(),'command':[PYTHON,'tools/run_dip_limit_study.py','--real'],'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'starting_worktree_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'automatic_retry':False,'attempt':1}
assert state['commit']=='66cd0b2866c603f8d88574e91b77e55f31cc2025' and state['starting_worktree_status']==''
assert not (ROOT/'reports/trend_dip_limit_round/results').exists()
with (AUDIT/'formal_attempt.json').open('x') as f:json.dump(state,f,indent=2)
print('Starting exactly one formal invocation from clean '+state['commit'],flush=True)
env=dict(os.environ,MPLCONFIGDIR=str(AUDIT/'matplotlib'),PYTHONDONTWRITEBYTECODE='1')
start=time.monotonic()
with (AUDIT/'formal_stdout.txt').open('x') as f:
    result=subprocess.run(state['command'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
state.update(ended_at=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-start,returncode=result.returncode,status='completed' if result.returncode==0 else 'STOPPED')
(AUDIT/'formal_attempt.json').write_text(json.dumps(state,indent=2)+'\n')
print((AUDIT/'formal_stdout.txt').read_text(),flush=True)
print(json.dumps(state,indent=2),flush=True)
sys.exit(result.returncode)

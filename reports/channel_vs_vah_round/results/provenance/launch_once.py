from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,traceback
root=Path('/home/ubuntu/.codex/worktrees/channel-vah-prereg/ppvp_quant')
log=Path('/tmp/channel_vah_run_570da83_n93t7yvq')
python='/home/ubuntu/Documents/Codex/2026-11-06/github-ppvp-quant-agent-2-checkpoint/work/diag_env/pandas3.0.5_py314/bin/python'
state={'command':[python,'-u','tools/run_breakout_study.py','--real'],'start_utc':datetime.now(timezone.utc).isoformat(),'automatic_retry':False,'run_attempt':1}
try:
    state['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    state['starting_worktree_status']=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
    assert state['commit']=='5f7b2e6fe24aa41b64035b98f07c9116341fd035'
    assert state['starting_worktree_status']==''
    assert not (root/'reports/channel_vs_vah_round/results').exists()
    (log/'formal_run_state.json').write_text(json.dumps(state,indent=2)+'\n')
    with (log/'formal_console.txt').open('x') as handle:
        proc=subprocess.Popen(state['command'],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:
            handle.write(line);handle.flush();print(line,end='',flush=True)
        status=proc.wait()
    state.update(exit_code=status,end_utc=datetime.now(timezone.utc).isoformat(),status='completed' if status==0 else 'STOPPED')
    (log/'formal_run_state.json').write_text(json.dumps(state,indent=2)+'\n')
    if status!=0:
        (log/'STOPPED_formal_run.json').write_text(json.dumps(state,indent=2)+'\n')
    raise SystemExit(status)
except Exception as exc:
    state.update(status='STOPPED',error=str(exc),traceback=traceback.format_exc(),end_utc=datetime.now(timezone.utc).isoformat())
    (log/'STOPPED_launcher.json').write_text(json.dumps(state,indent=2)+'\n')
    traceback.print_exc()
    raise SystemExit(1)

#!/usr/bin/env python3
"""Fixed CP3 preparation, registration and one-shot 87-account entrypoint."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import json
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def stopped(folder,name,stage,exc,state):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    target=folder/name
    if target.exists():return
    record={'status':'STOPPED','stage':stage,'time_utc':datetime.now(timezone.utc).isoformat(),
        'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
        'starting_state':state,'automatic_retry':False}
    with target.open('x') as f:json.dump(record,f,indent=2);f.write('\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--prepare-data',action='store_true')
    mode.add_argument('--register-implementation',action='store_true')
    mode.add_argument('--real',action='store_true')
    args=parser.parse_args(argv)
    from trend_hold_venue_study import scope,cp3_control as control,cp3_storage as storage
    if args.check:
        state=control.verify_frozen(ROOT,post_unlocked=scope.POST_IS_AND_ETH_UNLOCKED)
        print(json.dumps(state,indent=2));return 0
    if args.register_implementation:
        control.require_preparation()
        from trend_hold_venue_study.cp3_data import verify_prepared_inputs
        verify_prepared_inputs(ROOT)
        result=control.register_implementation(ROOT)
        print(json.dumps({'status':'implementation_registered','files':len(result['sha256'])}));return 0
    if args.prepare_data:
        control.require_preparation()
        state={}
        try:
            state['storage']=storage.verify_storage(ROOT)
            state['frozen']=control.verify_frozen(ROOT,post_unlocked=False)
            control.install_preparation_barrier()
            from trend_hold_venue_study.cp3_data import prepare_inputs,verify_prepared_inputs
            result=prepare_inputs(ROOT)
            evidence=verify_prepared_inputs(ROOT)
            print(json.dumps({'status':result['status'],'manifest':evidence['manifest'],
                              'source_and_cache_files_verified':len(evidence['files'])}));return 0
        except Exception as exc:
            stopped(ROOT/control.REG,'STOPPED_entrypoint_preparation.json','locked_input_preparation',exc,state)
            raise
    control.require_cp3()
    out=ROOT/control.OUT
    if out.exists():raise FileExistsError('CP3 output exists; no retry/resume')
    state={}
    try:
        state['storage']=storage.verify_storage(ROOT)
        state.update(control.verify_unlock(ROOT))
        from trend_hold_venue_study.cp3_data import load_inputs
        contexts=load_inputs(ROOT)
        state['input_registration']=json.loads((ROOT/control.INPUT_MANIFEST).read_text())
        from trend_hold_venue_study.cp3_runtime import execute
        result=execute(contexts,out,input_metadata=state)
        print(json.dumps({'status':result['status'],'accounts':result['combination_count'],
                          'figures':result['figures'],'stop_at':'CP3'}));return 0
    except Exception as exc:
        stopped(out,'STOPPED_entrypoint.json','formal_CP3',exc,state)
        raise


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)

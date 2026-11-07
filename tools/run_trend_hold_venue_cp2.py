#!/usr/bin/env python3
"""Fixed BTC IS CP2 entrypoint; raw preparation is a separate pre-unlock action."""
import argparse
from pathlib import Path
import sys,json,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--prepare-archives',action='store_true')
    mode.add_argument('--prepare-funding',action='store_true')
    mode.add_argument('--register-saved-funding',action='store_true')
    mode.add_argument('--real',action='store_true')
    args=parser.parse_args(argv)
    from trend_hold_venue_study import scope,cp2_control as control
    if args.prepare_archives:
        control.verify_frozen(ROOT,btc_unlocked=False)
        from trend_hold_venue_study.archives import prepare
        result=prepare(ROOT)
        print(json.dumps({'status':result['status'],'archives':len(result['archives'])}))
        return 0
    if args.prepare_funding:
        control.verify_frozen(ROOT,btc_unlocked=False)
        from trend_hold_venue_study.funding_sources import prepare
        result=prepare(ROOT)
        print(json.dumps({'status':result['status'],'common_settlements':result['common_settlements'],
            'overlap_settlements':result['overlap']['settlements'],
            'max_abs_decimal_diff':result['overlap']['max_abs_decimal_diff']}))
        return 0
    if args.register_saved_funding:
        control.verify_frozen(ROOT,btc_unlocked=False)
        from trend_hold_venue_study.funding_sources import register_saved
        result=register_saved(ROOT)
        print(json.dumps({'status':result['status'],'common_settlements':result['common_settlements'],
            'compared':result['overlap']['settlements'],'supplemented_overlap':result['supplemented_overlap_settlements'],
            'month_end_supplements':result['month_end_supplement_count']}))
        return 0
    if args.check:
        result=control.verify_frozen(ROOT,btc_unlocked=scope.BTC_IS_UNLOCKED)
        result['cp2_files']=len(control.verify_cp2(ROOT)['sha256'])+1
        print(json.dumps(result,sort_keys=True));return 0
    # Authorization precedes all real-input access; the other stage stays locked.
    control.require_btc_is()
    out=ROOT/control.OUT
    if out.exists():raise FileExistsError('formal output exists; no retry/resume')
    state={}
    try:
        state=control.verify_unlock(ROOT)
        from trend_hold_venue_study.data import load_inputs
        contexts,inputs=load_inputs(ROOT)
    except Exception as exc:
        out.mkdir(parents=True,exist_ok=False)
        (out/'STOPPED.json').write_text(json.dumps({'stage':'pre_match_input_verification','error_type':type(exc).__name__,
            'error':str(exc),'traceback':traceback.format_exc(),'starting_state':state,'automatic_retry':False},indent=2)+'\n')
        raise
    from trend_hold_venue_study.runtime import execute
    execute(contexts,out,{**inputs,**state})
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)

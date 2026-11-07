#!/usr/bin/env python3
"""Verify CP1 registration only; real execution is intentionally absent in CP1."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from trend_hold_venue_study.registration import verify
from trend_hold_venue_study import scope

if __name__=='__main__':
    if scope.BTC_IS_UNLOCKED or scope.POST_IS_AND_ETH_UNLOCKED:
        raise SystemExit('CP1 verification requires both stage locks False')
    print(json.dumps(verify(ROOT),sort_keys=True))

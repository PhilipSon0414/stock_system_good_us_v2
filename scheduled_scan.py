#!/usr/bin/env python3
"""Unattended entry point: compatible models first, then the existing daily report.

python scheduled_scan.py ALL          # run at the existing scheduled time
python scheduled_scan.py --prepare   # train missing/stale artifacts, do not scan/send
python scheduled_scan.py --check     # inspect readiness without writing or sending
"""
import argparse
import fcntl
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from market_clock import completed_session
from surge_model import load_model, MODEL_PATH

HERE = Path(__file__).resolve().parent

def model_ready():
    try:
        return load_model() is not None
    except (OSError, ValueError, EOFError, KeyError):
        return False

def stale(path, days):
    try:
        payload = load_model() if path == MODEL_PATH else json.loads(path.read_text())
        timestamp = payload.get('trained_at') or payload.get('generated_at')
        created = datetime.fromisoformat(timestamp)
        if created.tzinfo is None:
            return True  # legacy metadata has no reliable timezone
        age = (datetime.now(timezone.utc)-created).total_seconds()
        return age < -300 or age > days*86400
    except (OSError, ValueError, TypeError, AttributeError):
        return True

def rules_ready():
    try:
        payload = json.loads((HERE/'pattern_rules.json').read_text())
        return payload.get('schema_version') == 2
    except (OSError, ValueError):
        return False

def prepare():
    if not model_ready() or stale(MODEL_PATH, 7):
        subprocess.run([sys.executable, '-u', str(HERE/'surge_model.py'), 'train'], cwd=HERE, check=True)
    if not model_ready():
        raise RuntimeError('Compatible model was not produced; scan not published')
    if not rules_ready() or stale(HERE/'pattern_rules.json', 30):
        subprocess.run([sys.executable, '-u', str(HERE/'pattern_miner.py')], cwd=HERE, check=True)
    if not rules_ready():
        raise RuntimeError('Compatible pattern rules were not produced')

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('market', nargs='?', default='ALL', choices=['ALL','SP500','GROWTH'])
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--check', action='store_true')
    args=parser.parse_args(argv)
    if args.check:
        print(json.dumps({'session': completed_session(), 'model_ready': model_ready(),
                          'rules_ready': rules_ready(), 'python': sys.executable}, ensure_ascii=False))
        return
    logs=HERE/'logs'; logs.mkdir(exist_ok=True)
    with (logs/'scheduled.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another scheduled run is already active'); return
        if not args.prepare and (HERE/'reports'/f'{completed_session()}_scan.json').exists():
            print('Completed session already published; no duplicate email'); return
        prepare()
        if not args.prepare:
            import daily_scan
            daily_scan.main(args.market)

if __name__ == '__main__':
    main()

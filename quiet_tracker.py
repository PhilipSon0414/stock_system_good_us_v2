"""조용한 매집 신호 실측 추적.

일일 스캔의 조용한 매집 종목을 predictions/quiet_<날짜>.jsonl 에 기록하고,
20거래일이 지난 기록을 코칭과 같은 규칙(+12% 익절 / ATR 손절 / 20일 정리)으로 평가해
quiet_stats.json 에 전체·확증(✔)·미확증 그룹별로 집계한다.
(tracker.py 와 분리: 그쪽은 preds_*.jsonl 만 읽는다.)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import tracker
from config import COACH_TARGET
from data_fetcher import get_ohlcv
from execution import policy
from market_clock import session_dates, session_offset, completed_session

PRED_DIR = Path(__file__).parent / 'predictions' / 'v3'
STATS_PATH = Path(__file__).parent / 'quiet_stats_v3.json'
HORIZON = 20


def save_quiet(quiet: list[dict], scan_date: str) -> int:
    if not quiet:
        return 0
    PRED_DIR.mkdir(exist_ok=True, parents=True)
    path = PRED_DIR / f'quiet_{scan_date}.jsonl'
    if path.exists():
        return 0
    with open(path, 'x', encoding='utf-8') as f:
        for q in quiet:
            rec = {'scan_date': scan_date, 'ticker': q['ticker'], 'close': q['price'],
                   'vol_surge_3m': q['vol_surge_3m'], 'price_3m': q['price_3m'],
                   'box_pos': q.get('box_pos'), 'updn_vol': q.get('updn_vol'),
                   'confirmed': bool(q.get('confirmed')), 'atr_rel': q.get('atr_rel'),
                   'execution_policy': policy(q.get('atr_rel')),
                   'tech_score': q.get('tech_score'), 'prob': q.get('prob'), 'evals': {}}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return len(quiet)


def _group(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    n = len(rows)
    ev = [r['evals'][str(HORIZON)] for r in rows]
    return {'n': n,
            'hit_rate': round(sum(e['hit'] for e in ev) / n, 3),
            'win_rate': round(sum(e['rule_return'] > 0 for e in ev) / n, 3),
            'avg_return': round(sum(e['rule_return'] for e in ev) / n, 4),
            'target_rate': round(sum(e['rule_outcome'] == 'target' for e in ev) / n, 3),
            'stop_rate': round(sum(e['rule_outcome'] == 'stop' for e in ev) / n, 3),
            'time_rate': round(sum(e['rule_outcome'] == 'time' for e in ev) / n, 3)}


def evaluate(verbose: bool = True) -> dict | None:
    files = sorted(PRED_DIR.glob('quiet_*.jsonl'))
    if not files:
        return None
    cache: dict[str, pd.DataFrame] = {}
    recs: list[tuple[Path, list[dict]]] = []
    for p in files:
        recs.append((p, [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]))
    for p, rows in recs:
        changed = False
        for r in rows:
            if str(HORIZON) in r['evals']:
                continue
            t = r['ticker']
            if t not in cache:
                cache[t] = get_ohlcv(t, period='max')
            df = cache[t]
            df = df.loc[df.index <= pd.Timestamp(completed_session())]
            if df.empty:
                continue
            if pd.Timestamp(r['scan_date']) not in df.index:
                continue
            scale = r['close'] / float(df.loc[r['scan_date'], 'Close'])
            after = df[df.index > pd.Timestamp(r['scan_date'])].copy()
            after[['Open','High','Low','Close']] *= scale
            after = after.reindex(session_dates(session_offset(r['scan_date'], 1),
                                               session_offset(r['scan_date'], HORIZON)))
            rules = r['execution_policy']
            ev = tracker._eval_window(after, r['close'], HORIZON,
                                      rules['stop_pct'], rules['target'], rules)
            if ev is not None:
                r['evals'][str(HORIZON)] = ev
                changed = True
        if changed:
            p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    done = [r for _, rows in recs for r in rows if str(HORIZON) in r['evals']]
    stats = {'horizon': HORIZON, 'n_evaluated': len(done),
             'all': _group(done),
             'confirmed': _group([r for r in done if r['confirmed']]),
             'unconfirmed': _group([r for r in done if not r['confirmed']])}
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    if verbose:
        print(f'  [매집추적] {HORIZON}일 평가 {len(done)}건 '
              f'(확증 {(stats["confirmed"] or {}).get("n", 0)}건)')
    return stats

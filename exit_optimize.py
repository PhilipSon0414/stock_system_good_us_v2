"""계좌 수익률 기준 청산 규칙 탐색 — 앞 구간에서 고르고 뒤 구간에서 검증한다.

  python exit_optimize.py --cache PATH [--jobs 4]

vcp_backtest.py와 같은 재구성 엘리트 신호(walk-forward 표본 외 확률)를 쓰되,
건당 평균이 아니라 계좌 단위 결과로 비교한다:
  - 매일 시가에 신규 진입(합산점수 높은 순), 최대 PORTFOLIO_MAX_POSITIONS종목,
    비중 = 리스크 한도 / 손절폭, 총 투입 PORTFOLIO_MAX_DEPLOY 한도
  - 같은 날 청산금은 그날 재사용하지 않는다. 낙폭은 매일 종가 평가 기준.
  - 경로 운을 줄이려고 신호의 25%를 무작위로 뺀 반복 실행의 평균도 낸다.
선택 편향을 막기 위해 후보 규칙은 앞 구간(TRAIN)에서만 순위를 매기고,
뒤 구간(TEST)은 고른 규칙의 검증에만 쓴다. 생존 편향·재구성 한계는 vcp_backtest.py와 같다.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import vcp_strategy as vcp
from config import (SURGE_HORIZON, TECH_SCORE_GATE, FINAL_MIN_SCORE, W_SCORE, W_MODEL,
                    PORTFOLIO_MAX_POSITIONS, PORTFOLIO_RISK_PER_TRADE, PORTFOLIO_MAX_DEPLOY,
                    COACH_STOP_ATR_MULT, COACH_STOP_MIN, COACH_STOP_MAX)
from execution import simulate as execute, policy
from order_block import get_order_blocks
from scorer import score_technical
from universe import get_ticker_list
from vcp_backtest import load_data, build_frames, Existing, DEFAULT_CACHE

SPLIT = '2023-07-01'
OUT_JSON = Path(__file__).parent / 'reports' / 'exit_optimize.json'
_FRAMES: dict = {}


def _tech_job(args):
    t, idxs = args
    df = _FRAMES[t]
    return t, {int(i): score_technical(df.iloc[:i + 1], get_order_blocks(df.iloc[:i + 1]), info=None)[0]
               for i in idxs}


def elite_signals(frames, existing, tickers, start, cache: Path, jobs: int) -> list[dict]:
    """엘리트 조건을 통과한 모든 신호(종목 중복 포함 — 보유 중 여부는 계좌 시뮬레이션이 판단)."""
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    cands = {t: [(i, m) for i, m in existing.candidates(t, elite=True) if frames[t].index[i] >= start]
             for t in tickers}
    global _FRAMES
    _FRAMES = frames
    work = [(t, [i for i, _ in c]) for t, c in cands.items() if c]
    print(f'  기술점수 계산 {sum(len(w[1]) for w in work)}건')
    with ProcessPoolExecutor(jobs) as ex:
        tech = dict(ex.map(_tech_job, work))
    out = []
    for t, c in cands.items():
        for i, m in c:
            ts = tech[t][i]
            combined = round(W_SCORE * ts + W_MODEL * m['prob'] * 100)
            if ts >= TECH_SCORE_GATE and combined >= FINAL_MIN_SCORE:
                out.append({'ticker': t, 'i': i, 'prob': m['prob'], 'combined': combined})
    cache.write_bytes(pickle.dumps(out))
    return out


def atr_stop(atr: float) -> float:
    atr = atr if np.isfinite(atr) else 0.0
    return min(COACH_STOP_MAX, max(COACH_STOP_MIN, COACH_STOP_ATR_MULT * atr))


def make_trades(frames, arrays, signals, cfg) -> list[dict]:
    """신호별 매매 결과. e/j는 종목 일봉 위치, partial_j는 일부 익절 봉."""
    out = []
    for s in signals:
        t, i = s['ticker'], s['i']
        df = frames[t]
        o, h, l, c, ma10, ma20, atr = arrays[t]
        if i + 1 + SURGE_HORIZON > len(c):       # 두 규칙 모두 20일 미래가 있는 신호만
            continue
        if cfg['kind'] == 'existing':
            ev = execute(df.iloc[i + 1:i + 1 + SURGE_HORIZON], SURGE_HORIZON, policy(atr[i]))
            if ev is None:
                continue
            j = int(np.searchsorted(df.index.values, np.datetime64(ev['exit_date'])))
            out.append({**s, 'e': i + 1, 'j': j, 'ret': ev['rule_return'], 'entry': ev['entry_price'],
                        'stop_pct': atr_stop(atr[i]), 'partial_j': None, 'partial_ret': 0.0, 'frac': 0.0})
            continue
        p = {**vcp.PARAMS, **cfg['params']}
        if cfg['params'].get('stop_pct') == 'atr':
            p['stop_pct'] = atr_stop(atr[i])
        if p['partial_at'] is None:
            p['partial_at'], p['partial_frac'] = 9.0, 0.0
            p['trail_from_start'] = True
        e = i + 1
        entry = o[e] * (1 + p['slippage_bps'] / 1e4)
        ma = ma10 if p['trail_ma'] == 10 else ma20
        st = vcp.walk_exit(o, h, l, c, ma, e, entry, p)
        if st['closed']:
            j, ret = st['exit_index'], st['rule_return']
        else:                                     # 데이터 끝까지 보유 중 → 마지막 종가로 평가
            j, ret = len(c) - 1, st['open_return']
        out.append({**s, 'e': e, 'j': j, 'ret': float(ret), 'entry': float(entry),
                    'stop_pct': float(p['stop_pct']), 'partial_j': st['partial_index'],
                    'partial_ret': p['partial_frac'] * p['partial_at'], 'frac': p['partial_frac']})
    return out


def account_sim(trades, dates, pos_of, closes, max_weight=None, drop=None) -> dict:
    """일 단위 계좌 시뮬레이션. dates: 평가 거래일, pos_of[t][date]→일봉 위치."""
    by_day: dict = {}
    for k, t in enumerate(trades):
        if drop is not None and drop[k]:
            continue
        by_day.setdefault(t['entry_date'], []).append(t)
    equity, open_pos, curve, taken, exposure = 1.0, [], [], 0, 0.0
    for d in dates:
        still = []
        for p in open_pos:                        # 전일까지 청산된 포지션 실현
            if p['exit_date'] < d:
                equity += p['w'] * p['ret']
            else:
                still.append(p)
        open_pos = still
        for t in sorted(by_day.get(d, []), key=lambda x: (-x['combined'], x['ticker'])):
            if len(open_pos) >= PORTFOLIO_MAX_POSITIONS or any(p['ticker'] == t['ticker'] for p in open_pos):
                continue
            w = equity * PORTFOLIO_RISK_PER_TRADE / t['stop_pct']
            if max_weight is not None:
                w = min(w, equity * max_weight)
            w = min(w, equity * PORTFOLIO_MAX_DEPLOY - sum(p['w'] for p in open_pos))
            if w <= equity * 0.01:
                continue
            open_pos.append({**t, 'w': w})
            taken += 1
        mtm = equity
        for p in open_pos:
            if p['exit_date'] == d:
                mtm += p['w'] * p['ret']
                continue
            k = pos_of[p['ticker']].get(d)
            if k is None:
                continue
            done = p['partial_j'] is not None and p['partial_j'] <= k
            rem = 1 - p['frac'] if done else 1.0
            mtm += p['w'] * ((p['partial_ret'] if done else 0.0) + rem * (closes[p['ticker']][k] / p['entry'] - 1))
        exposure += sum(p['w'] for p in open_pos) / max(equity, 1e-9)
        curve.append(mtm)
    for p in open_pos:
        equity += p['w'] * p['ret']
    curve = np.array(curve + [equity])
    years = len(dates) / 252
    dd = float((curve / np.maximum.accumulate(curve) - 1).min())
    cagr = float(curve[-1] ** (1 / years) - 1) if curve[-1] > 0 else -1.0
    return {'cagr': round(cagr, 4), 'max_dd': round(dd, 4), 'n_taken': taken,
            'avg_exposure': round(exposure / len(dates), 3),
            'mar': round(cagr / -dd, 2) if dd < 0 else None}


def evaluate(trades, dates, pos_of, closes, max_weight, seeds: int = 30) -> dict:
    full = account_sim(trades, dates, pos_of, closes, max_weight)
    cagrs, dds = [], []
    for s in range(seeds):
        drop = np.random.default_rng(s).random(len(trades)) < 0.25
        r = account_sim(trades, dates, pos_of, closes, max_weight, drop)
        cagrs.append(r['cagr']); dds.append(r['max_dd'])
    return {**full, 'cagr_mean': round(float(np.mean(cagrs)), 4), 'cagr_p10': round(float(np.quantile(cagrs, .1)), 4),
            'max_dd_mean': round(float(np.mean(dds)), 4), '_cagrs': cagrs}


def label(cfg, mw) -> str:
    if cfg['kind'] == 'existing':
        base = '기존(+12%/ATR손절/20일)'
    else:
        p = cfg['params']
        stop = 'ATR' if p['stop_pct'] == 'atr' else f'{p["stop_pct"] * 100:.0f}%'
        part = '없음' if p['partial_at'] is None else f'{p["partial_at"] * 100:.0f}%'
        base = f'손절{stop}·절반익절{part}·MA{p["trail_ma"]}·최대{p["max_hold"]}일'
    return base + (f'·비중상한{mw * 100:.0f}%' if mw else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    ap.add_argument('--step', type=int, default=63)
    ap.add_argument('--jobs', type=int, default=4)
    args = ap.parse_args()

    data, spy = load_data(args.cache)
    frames, _ = build_frames(data)
    probs, _ = pickle.loads(args.cache.with_name(f'{args.cache.stem}_probs_{args.step}.pkl').read_bytes())
    start = probs['date'].min()
    tickers = [t for t in get_ticker_list('ALL') if t in frames]
    existing = Existing(frames, probs)
    signals = elite_signals(frames, existing, tickers, start,
                            args.cache.with_name(f'{args.cache.stem}_elite_{args.step}.pkl'), args.jobs)
    print(f'엘리트 신호 {len(signals)}건 ({start.date()}~)')

    arrays, pos_of, closes = {}, {}, {}
    for t in tickers:
        df = frames[t]
        c = df['Close']
        arrays[t] = (*(df[k].to_numpy(dtype=float) for k in ('Open', 'High', 'Low', 'Close')),
                     c.rolling(10).mean().to_numpy(), c.rolling(20).mean().to_numpy(), df['ATRRel'].to_numpy())
        pos_of[t] = {d: k for k, d in enumerate(df.index)}
        closes[t] = arrays[t][3]
    all_dates = [d for d in spy.index if d >= start]
    periods = {'train': [d for d in all_dates if d < pd.Timestamp(SPLIT)],
               'test': [d for d in all_dates if d >= pd.Timestamp(SPLIT)], 'all': all_dates}

    cfgs = [{'kind': 'existing'}]
    for stop, part, ma, hold in itertools.product((0.07, 0.10, 'atr'), (0.12, 0.15, 0.20, None), (10, 20), (40, 120)):
        cfgs.append({'kind': 'trend', 'params': {'stop_pct': stop, 'partial_at': part, 'partial_frac': 0.5,
                                                 'trail_ma': ma, 'max_hold': hold}})
    rows = []
    for cfg in cfgs:
        trades = make_trades(frames, arrays, signals, cfg)
        for t in trades:
            idx = frames[t['ticker']].index
            t['entry_date'], t['exit_date'] = idx[t['e']], idx[t['j']]
        for mw in (None, 0.10):
            row = {'label': label(cfg, mw), 'cfg': cfg, 'max_weight': mw}
            for name, dates in periods.items():
                sub = [t for t in trades if dates[0] <= t['entry_date'] <= dates[-1]]
                row[name] = evaluate(sub, dates, pos_of, closes, mw)
            rows.append(row)
            print(f"{row['label']:<46} train {row['train']['cagr_mean']*100:+5.1f}%/{row['train']['max_dd_mean']*100:5.1f}%"
                  f"  test {row['test']['cagr_mean']*100:+5.1f}%/{row['test']['max_dd_mean']*100:5.1f}%"
                  f"  all {row['all']['cagr']*100:+5.1f}%/{row['all']['max_dd']*100:5.1f}% n={row['all']['n_taken']}", flush=True)

    base = next(r for r in rows if r['cfg']['kind'] == 'existing' and r['max_weight'] is None)
    for r in rows:
        for name in periods:
            diff = np.array(r[name]['_cagrs']) - np.array(base[name]['_cagrs'])
            r[name]['beats_base'] = round(float((diff > 0).mean()), 2)
            r[name]['cagr_diff_mean'] = round(float(diff.mean()), 4)
    ranked = sorted(rows, key=lambda r: -r['train']['cagr_mean'])
    print('\n── 앞 구간(TRAIN) 평균 CAGR 상위 8개와 뒤 구간(TEST) 검증 ──')
    for r in ranked[:8] + [base]:
        print(f"{r['label']:<46} train {r['train']['cagr_mean']*100:+5.1f}% (기존 대비 승 {r['train']['beats_base']:.0%})"
              f"  test {r['test']['cagr_mean']*100:+5.1f}% (승 {r['test']['beats_base']:.0%}, 낙폭 {r['test']['max_dd_mean']*100:.1f}%)")
    tr = np.array([r['train']['cagr_mean'] for r in rows]); te = np.array([r['test']['cagr_mean'] for r in rows])
    print(f'\n규칙 {len(rows)}개의 TRAIN·TEST 순위 상관: {pd.Series(tr).corr(pd.Series(te), method="spearman"):.2f}')
    for r in rows:
        for name in periods:
            r[name].pop('_cagrs')
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps({'split': SPLIT, 'n_signals': len(signals), 'rows': rows,
                                    'ranked_by_train': [r['label'] for r in ranked]},
                                   ensure_ascii=False, indent=1, default=str))
    print(f'저장: {OUT_JSON}')


if __name__ == '__main__':
    main()

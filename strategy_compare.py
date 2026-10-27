"""Compare frozen prospective candidates using local OHLCV CSVs.

python strategy_compare.py --bars data/bars --output reports/comparison.json
CSV: Date,Open,High,Low,Close,Volume. Same adjusted price basis throughout.
No current model or current fundamentals are applied to historical signals.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from execution import simulate
from market_clock import session_dates, session_offset
from config import (SURGE_HORIZON, PORTFOLIO_MAX_POSITIONS, PORTFOLIO_MAX_DEPLOY,
                    PORTFOLIO_RISK_PER_TRADE, PORTFOLIO_MAX_CORR, TECH_SCORE_GATE)

VARIANTS = ('model', 'technical', 'mixed', 'mixed_elite')

def rank(rows, variant):
    rows = [r for r in rows if (r.get('pred') or {}).get('prob') is not None]
    if variant == 'mixed_elite':
        rows = [r for r in rows if r.get('elite_eligible', False)]
    field = 'tech_score' if variant == 'technical' else 'combined'
    return sorted(rows, key=lambda r: (-(r['pred']['prob'] if variant == 'model'
                                        else r[field]), r['ticker']))

def compare(snapshots, bars):
    prepared = {}
    skipped = []
    for snap in sorted(snapshots, key=lambda x:x['session']):
        day = snap['session']
        if not snap.get('model_id'):
            skipped.append({'session': day, 'reason': 'no validated model'}); continue
        dates = session_dates(session_offset(day, 1), session_offset(day, SURGE_HORIZON))
        # All strategies must see exactly the same complete market-data universe.
        trades = {}
        for r in snap['rows']:
            t = r['ticker']
            if t not in bars or t not in snap.get('execution_policies', {}):
                break
            rules = snap['execution_policies'][t]
            ev = simulate(bars[t].reindex(dates), SURGE_HORIZON, rules)
            if ev is None:
                break
            trades[t] = {**ev, 'rules': rules, 'ticker': t}
        else:
            prepared[dates[0]] = (snap, trades)
            continue
        skipped.append({'session': day, 'reason': 'incomplete horizon/OHLCV/policy'})
    if not prepared:
        return {'status': 'insufficient_data', 'skipped': skipped, 'variants': {}}
    start = min(prepared)
    end = max(pd.Timestamp(t['exit_date']) for _,trades in prepared.values() for t in trades.values())
    result = {}
    for variant in VARIANTS:
        cash, active, closed, curve = 1.0, {}, [], []
        for date in session_dates(start, end):
            open_value = lambda t: float(bars[t].loc[date,'Open'])
            equity = cash + sum(p['qty']*open_value(t) for t,p in active.items())
            if date in prepared:
                snap, trades = prepared[date]
                for r in rank(snap['rows'], variant):
                    t = r['ticker']
                    if t in active or len(active) >= PORTFOLIO_MAX_POSITIONS:
                        continue
                    # Correlation uses only data known before entry.
                    correlations = [bars[t].loc[bars[t].index < date,'Close'].pct_change().tail(60).corr(
                        bars[u].loc[bars[u].index < date,'Close'].pct_change().tail(60)) for u in active]
                    if any(not np.isfinite(c) or c >= PORTFOLIO_MAX_CORR for c in correlations):
                        continue
                    trade = trades[t]
                    deployed = sum(p['qty']*open_value(u) for u,p in active.items())
                    risk_used = sum(p['capital']*p['rules']['stop_pct'] for p in active.values())
                    risk_room = max(0, equity*PORTFOLIO_RISK_PER_TRADE*PORTFOLIO_MAX_POSITIONS-risk_used)
                    capital = min(cash, max(0, equity*PORTFOLIO_MAX_DEPLOY-deployed),
                                  equity*PORTFOLIO_RISK_PER_TRADE/trade['rules']['stop_pct'],
                                  risk_room/trade['rules']['stop_pct'])
                    if capital <= 1e-10:
                        continue
                    qty = capital/(trade['entry_price']*(1+trade['rules']['fee_bps']/10000))
                    active[t] = {**trade, 'capital': capital, 'qty': qty}
                    cash -= capital
            # Conservatively do not recycle same-day exit proceeds into open entries.
            for t,p in list(active.items()):
                if pd.Timestamp(p['exit_date']) == date:
                    cash += p['qty']*p['exit_price']*(1-p['rules']['fee_bps']/10000)
                    closed.append(p); del active[t]
            nav = cash + sum(p['qty']*float(bars[t].loc[date,'Close']) for t,p in active.items())
            curve.append(nav)
        nav = np.asarray(curve)
        returns = [p['rule_return'] for p in closed]
        result[variant] = {'n_trades': len(closed), 'net_return': float(nav[-1]-1),
                           'max_drawdown': float((nav/np.maximum.accumulate(np.r_[1,nav])[1:]-1).min()),
                           'win_rate': float(np.mean(np.array(returns)>0)) if returns else None,
                           'mean_trade_return': float(np.mean(returns)) if returns else None,
                           'equity': curve}
    return {'status': 'ok', 'skipped': skipped, 'variants': result,
            'note': 'Prospective frozen signals, cost assumptions, portfolio caps; not broker returns. No regime overlay; filters use frozen model threshold.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', default='reports/candidates')
    parser.add_argument('--bars', required=True)
    parser.add_argument('--output', default='reports/comparison.json')
    args = parser.parse_args()
    snapshots = [json.loads(p.read_text()) for p in Path(args.snapshots).glob('*.json')]
    tickers = {r['ticker'] for s in snapshots for r in s['rows']}
    bars = {t: pd.read_csv(Path(args.bars)/f'{t}.csv', parse_dates=['Date']).set_index('Date').sort_index()
            for t in tickers if (Path(args.bars)/f'{t}.csv').exists()}
    out = compare(snapshots, bars)
    dest = Path(args.output); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f'{out["status"]}: {dest}')

if __name__ == '__main__':
    main()

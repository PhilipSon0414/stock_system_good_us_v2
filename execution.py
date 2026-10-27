"""Versioned, next-session-open paper execution. Not brokerage fills."""
import numpy as np
from config import (COACH_TARGET, COACH_STOP_MIN, COACH_STOP_MAX,
                    COACH_STOP_ATR_MULT, COACH_D3_CUT_MIN, COACH_D3_CUT_ATR)

VERSION = 'next_open_v1'

def policy(atr=None):
    atr = float(atr) if atr is not None and np.isfinite(atr) else 0.0
    return {'version': VERSION, 'entry': 'next_open', 'target': COACH_TARGET,
            'stop_pct': min(COACH_STOP_MAX, max(COACH_STOP_MIN, COACH_STOP_ATR_MULT*atr)),
            'd3_cut_pct': max(COACH_D3_CUT_MIN, COACH_D3_CUT_ATR*atr),
            'fee_bps': 5.0, 'slippage_bps': 10.0}

def simulate(after, horizon, rules):
    """D3 close exit signal fills at D4 open. Same-day ambiguity: stop first.

    Price barriers use entry fill. Limit exits get no positive gap improvement.
    Costs are assumptions saved with each signal, not claimed broker costs.
    """
    win = after.head(horizon)
    if len(win) < horizon:
        return None
    bars = win[['Open','High','Low','Close']].to_numpy(dtype=float)
    if not np.isfinite(bars).all() or (bars <= 0).any():
        return None
    if ((bars[:,1] < np.maximum(bars[:,0],bars[:,3])) |
        (bars[:,2] > np.minimum(bars[:,0],bars[:,3]))).any():
        return None
    slip, fee = rules['slippage_bps']/10000, rules['fee_bps']/10000
    entry = bars[0,0]*(1+slip)
    stop, target = entry*(1-rules['stop_pct']), entry*(1+rules['target'])
    exit_price, reason, exit_i = bars[-1,3]*(1-slip), 'time', horizon-1
    cut_pending = False
    for i,(o,h,l,c) in enumerate(bars):
        if o <= stop:
            exit_price,reason,exit_i = o*(1-slip),'stop',i; break
        if o >= target:
            exit_price,reason,exit_i = target,'target',i; break
        if cut_pending:
            exit_price,reason,exit_i = o*(1-slip),'d3',i; break
        if l <= stop:
            exit_price,reason,exit_i = stop*(1-slip),'stop',i; break
        if h >= target:
            exit_price,reason,exit_i = target,'target',i; break
        if i == 2 and c/entry-1 <= -rules['d3_cut_pct']:
            cut_pending = True
    return {'entry_price': float(entry), 'exit_price': float(exit_price),
            'entry_date': str(win.index[0].date()), 'exit_date': str(win.index[exit_i].date()),
            'rule_return': round(float(exit_price*(1-fee)/(entry*(1+fee))-1), 6),
            'rule_outcome': reason, 'execution_version': rules['version']}

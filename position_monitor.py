"""보유·추천 종목 일일 점검 — "오늘 무엇을 해야 하는가"를 종목별로 낸다.

청산 규칙(trend_v1, config의 TREND_* 값):
  - 진입 즉시 손절 주문: 체결가 -7%
  - +15% 도달 시 절반 익절(지정가), 남은 물량의 손절을 본전(체결가)으로 상향
  - 절반 익절 뒤에는 종가가 10일 이동평균선(TREND_TRAIL_MA) 아래로 내려가면 다음 거래일 시가에 전량 매도
  - 그 전까지는 주문을 건 채로 두면 되고 매일 할 일은 없다

두 가지를 점검한다:
  1. 추천 추적: predictions/v3의 과거 엘리트픽을 다음 거래일 시가에 샀다고 가정하고 진행 상태를 낸다.
     청산이 끝난 건은 trend_stats_v3.json에 집계한다(모의 결과, 계좌 체결 아님).
  2. 내 보유: portfolio_state.json의 positions 중 entry_date·entry_price가 있는 종목.
     (환경변수 PORTFOLIO_POSITIONS에 같은 형식의 JSON 목록을 넣어도 된다)

규칙 로직은 vcp_strategy.walk_exit을 그대로 써서 백테스트와 일일 점검이 같은 판정을 한다.
주문은 제출하지 않는다.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import vcp_strategy as vcp
from config import (TREND_STOP_PCT, TREND_PARTIAL_AT, TREND_PARTIAL_FRAC,
                    TREND_TRAIL_MA, TREND_MAX_HOLD)
from data_fetcher import get_ohlcv
from market_clock import completed_bars
from storage import write_text

HERE = Path(__file__).parent
PRED_DIR = HERE / 'predictions' / 'v3'
STATS_PATH = HERE / 'trend_stats_v3.json'
STATE_PATH = HERE / 'portfolio_state.json'
VERSION = 'trend_v1'
RULE = {**vcp.PARAMS, 'stop_pct': TREND_STOP_PCT, 'partial_at': TREND_PARTIAL_AT,
        'partial_frac': TREND_PARTIAL_FRAC, 'trail_ma': TREND_TRAIL_MA,
        'trail_from_start': False, 'max_hold': TREND_MAX_HOLD}

ACTIONS = {
    'hold': '조치 없음',
    'raise_stop': '절반 익절 체결 확인 → 손절 주문을 본전으로 변경',
    'sell_next_open': '다음 거래일 시가에 전량 매도',
    'closed': '규칙상 청산 완료',
    'pending_entry': '다음 거래일 시가 진입 대기',
}
REASONS = {'stop': '손절', 'breakeven': '절반 익절 후 본전 청산', 'trail': f'절반 익절 후 {TREND_TRAIL_MA}일선 이탈',
           'time': '최대 보유 기한'}


def status(df: pd.DataFrame, entry_date: str, entry_price: float,
           from_open: bool, rule: dict = RULE) -> dict:
    """entry_date에 entry_price로 산 포지션의 현재 상태와 다음 거래일 행동.
    from_open=False(실제 보유)는 체결 시각을 모르므로 진입 당일 고가·저가로는 판정하지 않는다."""
    idx = df.index
    e = int(np.searchsorted(idx.values, np.datetime64(pd.Timestamp(entry_date))))
    base = {'entry_date': str(pd.Timestamp(entry_date).date()), 'entry_price': round(float(entry_price), 2),
            'rule': VERSION}
    if e >= len(df):
        return {**base, 'action': 'pending_entry', 'action_text': ACTIONS['pending_entry']}
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ('Open', 'High', 'Low', 'Close'))
    ma = df['Close'].rolling(rule['trail_ma']).mean().to_numpy()
    st = vcp.walk_exit(o, h, l, c, ma, e, float(entry_price), rule, first_barrier=e if from_open else e + 1)
    partial_date = str(idx[st['partial_index']].date()) if st['partial_index'] is not None else None
    base.update({'partial_done': st['partial'], 'partial_date': partial_date,
                 'max_gain_pct': round(st['max_gain'] * 100, 1)})
    if st['closed']:
        j = st['exit_index']
        return {**base, 'action': 'closed', 'action_text': ACTIONS['closed'],
                'exit_date': str(idx[j].date()), 'exit_reason': st['rule_outcome'],
                'exit_reason_text': REASONS[st['rule_outcome']],
                'return_pct': round(st['rule_return'] * 100, 2), 'days_held': int(j - e + 1)}
    last = st['last_index']
    if st['pending']:
        action = 'sell_next_open'
    elif st['partial'] and st['partial_index'] == last:
        action = 'raise_stop'
    else:
        action = 'hold'
    out = {**base, 'action': action, 'action_text': ACTIONS[action],
           'pending_reason': REASONS.get(st['pending']) if st['pending'] else None,
           'days_held': int(last - e + 1), 'last_close': round(float(c[last]), 2),
           'return_pct': round(st['open_return'] * 100, 2),
           'price_change_pct': round((c[last] / entry_price - 1) * 100, 2),
           'stop': round(st['stop'], 2),
           'stop_basis': '본전' if st['partial'] else f'-{rule["stop_pct"] * 100:.0f}%',
           'partial_target': None if st['partial'] else round(st['trigger'], 2),
           'ma': round(float(ma[last]), 2) if np.isfinite(ma[last]) else None,
           'remaining_fraction': round(st['remaining'], 2)}
    out['orders'] = _orders(out, rule)
    return out


def _orders(s: dict, rule: dict) -> str:
    """내일 장 전에 걸려 있어야 하는 주문 상태를 한 줄로."""
    if s['action'] == 'sell_next_open':
        return f'시가 전량 매도 ({s["pending_reason"]}). 남은 손절 주문은 취소'
    if s['partial_done']:
        return (f'손절 ${s["stop"]:.2f}(본전) 유지. 종가가 {rule["trail_ma"]}일선'
                f'(${s["ma"]:.2f}) 아래로 마감하면 다음 날 시가 매도')
    return (f'손절 ${s["stop"]:.2f} 유지, ${s["partial_target"]:.2f}에 절반 매도 지정가 유지')


def _bars(ticker: str, session: str, cache: dict) -> pd.DataFrame | None:
    if ticker not in cache:
        try:
            cache[ticker] = completed_bars(get_ohlcv(ticker, period='2y', end_date=session), session)
        except (ValueError, KeyError):
            cache[ticker] = None
    return cache[ticker]


# ── 추천 추적 ────────────────────────────────────────────────────────────────

def _elite_history(session: str) -> list[dict]:
    rows = []
    for path in sorted(PRED_DIR.glob('preds_*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get('elite') and r['scan_date'] <= session:
                    rows.append(r)
    return sorted(rows, key=lambda r: (r['scan_date'], r['ticker']))


def track_recommendations(session: str, cache: dict | None = None, rule: dict = RULE) -> dict:
    """과거 엘리트픽을 다음 거래일 시가(+슬리피지)에 샀다고 보고 규칙대로 진행.
    같은 종목은 앞선 추천이 청산되기 전의 재추천을 새 진입으로 세지 않는다."""
    cache = {} if cache is None else cache
    open_rows, closed_rows, failed, busy = [], [], [], {}
    for r in _elite_history(session):
        t = r['ticker']
        if t in busy and r['scan_date'] <= busy[t]:
            continue
        df = _bars(t, session, cache)
        signal = pd.Timestamp(r['scan_date'])
        if df is None or signal not in df.index:
            failed.append(t)
            continue
        # 배당·분할로 수정주가가 바뀌어도 기록된 신호 종가 기준으로 맞춘다
        scale = r['close'] / float(df.at[signal, 'Close'])
        after = df.index[df.index > signal]
        row = {'ticker': t, 'signal_date': r['scan_date'], 'signal_close': round(r['close'], 2),
               'prob': r.get('prob'), 'combined': r.get('combined')}
        if len(after) == 0:
            open_rows.append({**row, 'action': 'pending_entry', 'action_text': ACTIONS['pending_entry'], 'rule': VERSION})
            busy[t] = '9999-12-31'
            continue
        scaled = df.copy()
        scaled[['Open', 'High', 'Low', 'Close']] *= scale
        entry = float(scaled.at[after[0], 'Open']) * (1 + rule['slippage_bps'] / 1e4)
        st = status(scaled, str(after[0].date()), entry, from_open=True, rule=rule)
        if st['action'] == 'closed':
            closed_rows.append({**row, **st})
            busy[t] = st['exit_date']
        else:
            open_rows.append({**row, **st})
            busy[t] = '9999-12-31'
    return {'open': open_rows, 'closed': closed_rows, 'failed': sorted(set(failed))}


def summarize_closed(closed: list[dict]) -> dict | None:
    if not closed:
        return None
    r = np.array([c['return_pct'] for c in closed]) / 100
    wins, losses = r[r > 0], r[r <= 0]
    outcomes = pd.Series([c['exit_reason'] for c in closed]).value_counts(normalize=True)
    return {'n': len(r), 'win_rate': round(float((r > 0).mean()), 3),
            'avg_return': round(float(r.mean()), 4),
            'avg_win': round(float(wins.mean()), 4) if len(wins) else None,
            'avg_loss': round(float(losses.mean()), 4) if len(losses) else None,
            'avg_hold_days': round(float(np.mean([c['days_held'] for c in closed])), 1),
            'outcomes': {k: round(float(v), 3) for k, v in outcomes.items()}}


# ── 내 보유 ──────────────────────────────────────────────────────────────────

def load_holdings(path: Path = STATE_PATH) -> tuple[list[dict], str | None]:
    """entry_date·entry_price가 있는 보유 종목. as_of_session 일치는 요구하지 않는다
    (체결 기록은 날짜가 지나도 유효하다). 비중 제안용 load_state와는 별개."""
    raw = os.environ.get('PORTFOLIO_POSITIONS')
    try:
        if raw:
            positions = json.loads(raw)
        elif path.exists():
            positions = json.loads(path.read_text(encoding='utf-8')).get('positions', [])
        else:
            return [], '보유 종목 미등록: 내 보유 점검 생략 (추천 추적만 표시)'
        out = []
        for p in positions:
            if 'entry_date' not in p or 'entry_price' not in p:
                continue
            price = float(p['entry_price'])
            if not (np.isfinite(price) and price > 0):
                raise ValueError('invalid entry_price')
            out.append({'ticker': str(p['ticker']).upper(),
                        'entry_date': str(pd.Timestamp(p['entry_date']).date()),
                        'entry_price': price, 'shares': p.get('shares')})
        if not out:
            return [], '보유 종목에 entry_date·entry_price가 없어 내 보유 점검 생략'
        return out, None
    except (ValueError, TypeError, KeyError, AttributeError):
        return [], '보유 종목 형식 오류: 내 보유 점검 생략'


def track_holdings(session: str, cache: dict | None = None, rule: dict = RULE) -> dict:
    cache = {} if cache is None else cache
    holdings, note = load_holdings()
    rows = []
    for p in holdings:
        df = _bars(p['ticker'], session, cache)
        if df is None:
            rows.append({**p, 'action': 'unknown', 'action_text': '시세 확인 실패 — 점검 불가'})
            continue
        st = status(df, p['entry_date'], p['entry_price'], from_open=False, rule=rule)
        if st['action'] == 'closed':
            st['action_text'] = (f'규칙상 {st["exit_date"]}에 청산됐어야 함 ({st["exit_reason_text"]})'
                                 ' — 아직 보유 중이면 다음 거래일 시가 정리')
        rows.append({**p, **st})
    return {'positions': rows, 'note': note}


def run(session: str, save_stats: bool = True) -> dict:
    cache: dict = {}
    recs = track_recommendations(session, cache)
    held = track_holdings(session, cache)
    stats = summarize_closed(recs['closed'])
    if save_stats:
        write_text(STATS_PATH, json.dumps({
            'updated_session': session, 'rule': VERSION,
            'rule_params': {k: RULE[k] for k in ('stop_pct', 'partial_at', 'partial_frac', 'trail_ma', 'max_hold')},
            'entry': 'next_open', 'closed': stats, 'n_open': len(recs['open']),
            'closed_trades': recs['closed']}, ensure_ascii=False, indent=2))
    recent = sorted(recs['closed'], key=lambda c: c['exit_date'], reverse=True)[:5]
    return {'rule': VERSION, 'session': session, 'holdings': held['positions'],
            'holdings_note': held['note'], 'recommendations_open': recs['open'],
            'recently_closed': recent, 'closed_stats': stats, 'failed': recs['failed']}


def report_lines(tr: dict, sep2: str) -> list[str]:
    L = ['  [ 보유·추천 종목 점검 — 내일 할 일 ]', sep2,
         f'  규칙: 손절 -{RULE["stop_pct"] * 100:.0f}% → +{RULE["partial_at"] * 100:.0f}%에서 절반 익절·손절 본전'
         f' → 이후 {RULE["trail_ma"]}일선 종가 이탈 시 다음 시가 매도', sep2]

    def line(r: dict) -> str:
        head = f'  ▸ {r["ticker"]:<6}'
        if r['action'] == 'pending_entry':
            return f'{head} {r["action_text"]}'
        if r['action'] == 'unknown':
            return f'{head} {r["action_text"]}'
        if r['action'] == 'closed':
            return (f'{head} {r["action_text"]} | 진입 {r["entry_date"]} ${r["entry_price"]:.2f}'
                    f' → {r["exit_date"]} {r["exit_reason_text"]} {r["return_pct"]:+.1f}%')
        return (f'{head} D+{r["days_held"]} ${r["last_close"]:.2f} ({r["return_pct"]:+.1f}%)'
                f' | {r["action_text"]} | {r["orders"]}')

    L.append('  내 보유')
    if tr['holdings']:
        L += [line(r) for r in tr['holdings']]
    else:
        L.append(f'    {tr["holdings_note"] or "보유 종목 없음"}')
    L.append('  추천 추적 (과거 엘리트픽을 다음 시가에 샀다고 가정)')
    if tr['recommendations_open']:
        L += [line(r) for r in tr['recommendations_open']]
    else:
        L.append('    진행 중인 추천 없음')
    for r in tr['recently_closed']:
        L.append(f'    최근 청산: {r["ticker"]} {r["exit_date"]} {r["exit_reason_text"]} {r["return_pct"]:+.1f}%')
    s = tr['closed_stats']
    if s:
        L.append(f'  새 규칙 모의 성과: 청산 {s["n"]}건, 승률 {s["win_rate"] * 100:.0f}%,'
                 f' 평균 {s["avg_return"] * 100:+.1f}% (계좌 체결 아님)')
    else:
        L.append('  새 규칙 모의 성과: 청산 완료 건 없음 — 평가 대기')
    if tr['failed']:
        L.append(f'  시세 확인 실패: {", ".join(tr["failed"])}')
    L.append(sep2)
    return L

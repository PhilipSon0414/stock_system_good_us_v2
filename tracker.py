"""
예측 추적기 — v2 (v1의 존재하지 않던 stage_tracker를 리포 내부 모듈로 구현)

역할:
  1. save_predictions(): 매 스캔의 픽(티커/종가/확률/점수)을 predictions/에 JSONL로 기록
  2. evaluate(): 기한이 지난 예측을 HOLD_HORIZONS(10일/20일)별로 실측 대조.
     보유 방식(매수 후 2~4주 관찰 → 매도)에 맞춰 세 지표를 모두 기록한다:
       - max_gain   : 기간 내 고가 기준 최대 상승 (+10% 지정가 매도가 체결됐을지)
       - end_return : 기간 마지막 날 종가 수익률 (기한까지 들고 있다 팔았을 때)
       - max_dd     : 기간 내 저가 기준 최대 낙폭 (손절선 설계 근거)
       - rule_*     : 코칭 규칙(+10% GTC / ATR 손절 / 기한 정리)대로 매매했을 때의
                      결과. hit(터치율)은 손절을 무시하므로 실제 승률과 다르다 —
                      사용자에게 보여줄 "적중률"은 rule.win_rate 쪽이다.
     결과는 tracker_stats.json에 horizon별로 갱신.
  3. recommended_min_prob(): 주 기한(SURGE_HORIZON)의 실현 정밀도가 목표에
     도달하는 최소 확률 구간 반환 (다음 스캔의 게이트로 사용)

predictions/ 와 tracker_stats.json 은 git에 커밋된다 — 학습 이력이
리포지토리만으로 재현 가능해야 한다 (v1은 전부 .gitignore에 걸려 있었다).

레코드 형식: 구형(단일 evaluated/hit/max_gain, 10일 기준)은 로드 시
evals.h10으로 자동 이관되며, end_return/max_dd는 다음 evaluate()에서 백필된다.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from config import (SURGE_TARGET, SURGE_HORIZON, HOLD_HORIZONS, TARGET_PRECISION,
                    COACH_STOP_MIN, TRACKER_MIN_BUCKET_N, RECENT_SCANS)
from data_fetcher import get_ohlcv

HERE = Path(__file__).parent
PRED_DIR   = HERE / 'predictions'
STATS_PATH = HERE / 'tracker_stats.json'
PRED_DIR.mkdir(exist_ok=True)

# 20일 라벨은 보정 확률이 높게 분포하므로 상단(0.65) 버킷을 추가해 변별
PROB_BUCKETS = [0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 1.01]

RULE_OUTCOMES = ('target', 'stop', 'time')


def save_predictions(rows: list[dict], scan_date: str) -> int:
    """rows: [{ticker, close, prob, tech_score, combined, atr_rel, stop_pct}, ...]"""
    path = PRED_DIR / f'preds_{scan_date}.jsonl'
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            rec = {'scan_date': scan_date, 'model_horizon': SURGE_HORIZON,
                   'evals': {}, **r}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return len(rows)


def _migrate(r: dict) -> dict:
    """구형 레코드(evaluated/hit/max_gain 단일 평가, 10일 기준) → evals.h10."""
    if 'evals' not in r:
        r['evals'] = {}
        if r.pop('evaluated', False):
            r['evals']['h10'] = {'max_gain': r.pop('max_gain', None),
                                 'hit': r.pop('hit', None)}
    return r


def _load_all() -> list[dict]:
    out = []
    for p in sorted(PRED_DIR.glob('preds_*.jsonl')):
        with open(p, encoding='utf-8') as f:
            out += [_migrate(json.loads(line)) for line in f if line.strip()]
    return out


def _write_back(records: list[dict]) -> None:
    by_date: dict[str, list[dict]] = {}
    for r in records:
        by_date.setdefault(r['scan_date'], []).append(r)
    for d, rows in by_date.items():
        with open(PRED_DIR / f'preds_{d}.jsonl', 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')


def _eval_window(after, entry_close: float, horizon: int,
                 stop_pct: float) -> dict | None:
    """scan_date 이후 horizon 거래일 창의 실측 지표. 창이 덜 찼으면 None.

    hit          : 창 내 고가가 +SURGE_TARGET 터치 (손절 무시 — 모델 라벨과 동일)
    rule_outcome : 코칭 규칙대로 매매했을 때의 결과 —
                   D0 종가 진입, +TARGET GTC 지정가, -stop_pct 손절, 기한 종가 정리.
                   같은 날 목표·손절이 모두 걸리면 손절로 간주(보수적).
    """
    win = after.head(horizon)
    if len(win) < horizon:
        return None
    high = win['High'].values / entry_close - 1
    low  = (win['Low'] if 'Low' in win.columns else win['Close']).values / entry_close - 1
    close = win['Close'].values / entry_close - 1

    hit_days  = np.where(high >= SURGE_TARGET)[0]
    stop_days = np.where(low <= -stop_pct)[0]
    first_hit  = int(hit_days[0]) + 1 if len(hit_days) else None
    first_stop = int(stop_days[0]) + 1 if len(stop_days) else None

    if first_stop is not None and (first_hit is None or first_stop <= first_hit):
        outcome, rule_ret = 'stop', -stop_pct
    elif first_hit is not None:
        outcome, rule_ret = 'target', SURGE_TARGET
    else:
        outcome, rule_ret = 'time', float(close[-1])

    return {'max_gain':   round(float(high.max()), 4),
            'end_return': round(float(close[-1]), 4),
            'max_dd':     round(float(low.min()), 4),
            'hit':        bool(high.max() >= SURGE_TARGET),
            'first_hit_day':  first_hit,
            'first_stop_day': first_stop,
            'd3_close':   round(float(close[2]), 4) if len(close) >= 3 else None,
            'stop_pct':   stop_pct,
            'rule_outcome': outcome,
            'rule_return':  round(float(rule_ret), 4)}


def _needs_eval(r: dict, horizon: int) -> bool:
    ev = r['evals'].get(f'h{horizon}')
    # 구형 평가(규칙 결과 없음)도 재평가해 백필
    return ev is None or 'rule_outcome' not in ev


def _independent(done: list[dict], horizon: int) -> list[dict]:
    """종목별로 평가 창(horizon 거래일 ≈ 1.4배 달력일)이 겹치지 않는 레코드만.
    연일 반복 등장하는 종목이 같은 급등을 여러 번 세는 것을 막는다."""
    gap = timedelta(days=int(horizon * 1.4))
    last: dict[str, datetime] = {}
    out = []
    for r in sorted(done, key=lambda r: (r['ticker'], r['scan_date'])):
        d = datetime.strptime(r['scan_date'], '%Y-%m-%d')
        if r['ticker'] not in last or d - last[r['ticker']] >= gap:
            out.append(r)
            last[r['ticker']] = d
    return out


def _buckets(done: list[dict], horizon: int) -> list[dict]:
    hits  = np.array([bool(r['evals'][f'h{horizon}']['hit']) for r in done])
    probs = np.array([r.get('prob') if r.get('prob') is not None else -1 for r in done])
    out = []
    for lo, hi in zip(PROB_BUCKETS[:-1], PROB_BUCKETS[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum():
            out.append({'prob_min': lo, 'prob_max': hi, 'n': int(m.sum()),
                        'hit_rate': round(float(hits[m].mean()), 3)})
    return out


def _recent(records: list[dict], horizon: int = 10) -> dict | None:
    """최근 RECENT_SCANS회 스캔(평가 완료분)의 실측 — 국면 악화 조기 감지용."""
    done = [r for r in records if r['evals'].get(f'h{horizon}')]
    dates = sorted({r['scan_date'] for r in done})[-RECENT_SCANS:]
    if not dates:
        return None
    sub = [r for r in done if r['scan_date'] in dates]
    rule_ret = np.array([r['evals'][f'h{horizon}']['rule_return'] for r in sub])
    return {'horizon': horizon, 'n_scans': len(dates), 'n': len(sub),
            'from': dates[0], 'to': dates[-1],
            'hit_rate': round(float(np.mean([r['evals'][f'h{horizon}']['hit'] for r in sub])), 3),
            'rule_win_rate': round(float((rule_ret > 0).mean()), 3),
            'rule_avg_return': round(float(rule_ret.mean()), 4)}


def evaluate(verbose: bool = True) -> dict | None:
    """기한이 지난 미평가 예측을 HOLD_HORIZONS별로 실측과 대조. 통계 갱신."""
    records = _load_all()
    if not records:
        return None

    # 최소 기한(거래일) ≈ 1.4배 달력일 — 그 전엔 fetch 자체를 생략
    prefilter = datetime.now() - timedelta(days=int(min(HOLD_HORIZONS) * 1.4) + 2)
    cache: dict = {}

    for r in records:
        pending = [h for h in HOLD_HORIZONS if _needs_eval(r, h)]
        if not pending:
            continue
        if datetime.strptime(r['scan_date'], '%Y-%m-%d') > prefilter:
            continue
        if r['ticker'] not in cache:
            cache[r['ticker']] = get_ohlcv(r['ticker'], period='6mo')
        df = cache[r['ticker']]
        if df.empty:
            continue
        after = df[df.index > r['scan_date']]
        # 구형 레코드(ATR 손절 도입 전)는 당시 규칙이던 고정 하한 손절로 평가
        stop_pct = r.get('stop_pct') or COACH_STOP_MIN
        for h in pending:
            ev = _eval_window(after, r['close'], h, stop_pct)
            if ev is not None:
                r['evals'][f'h{h}'] = ev

    _write_back(records)

    stats = {'updated_at': datetime.now().isoformat(),
             'target': SURGE_TARGET,
             'primary_horizon': SURGE_HORIZON,
             'horizons': {}}

    for h in HOLD_HORIZONS:
        done = [r for r in records if r['evals'].get(f'h{h}')]
        if not done:
            continue
        hits  = np.array([bool(r['evals'][f'h{h}']['hit']) for r in done])
        ends  = np.array([r['evals'][f'h{h}'].get('end_return')
                          if r['evals'][f'h{h}'].get('end_return') is not None
                          else np.nan for r in done], dtype=float)
        dds   = np.array([r['evals'][f'h{h}'].get('max_dd')
                          if r['evals'][f'h{h}'].get('max_dd') is not None
                          else np.nan for r in done], dtype=float)
        indep = _independent(done, h)

        outcomes = np.array([r['evals'][f'h{h}']['rule_outcome'] for r in done])
        rule_ret = np.array([r['evals'][f'h{h}']['rule_return'] for r in done], dtype=float)
        stopped_winners = np.array([
            r['evals'][f'h{h}']['hit'] and r['evals'][f'h{h}']['rule_outcome'] == 'stop'
            for r in done])

        stats['horizons'][str(h)] = {
            'n_evaluated': len(done),
            'hit_rate': round(float(hits.mean()), 3),
            'avg_end_return':    (round(float(np.nanmean(ends)), 4)
                                  if np.isfinite(ends).any() else None),
            'median_end_return': (round(float(np.nanmedian(ends)), 4)
                                  if np.isfinite(ends).any() else None),
            'avg_max_dd':        (round(float(np.nanmean(dds)), 4)
                                  if np.isfinite(dds).any() else None),
            # 코칭 규칙 기준 (목표 GTC / 손절 / 기한 정리) — 실제 매매 승률
            'rule': {
                'win_rate':   round(float((rule_ret > 0).mean()), 3),
                'avg_return': round(float(rule_ret.mean()), 4),
                **{f'{o}_rate': round(float((outcomes == o).mean()), 3)
                   for o in RULE_OUTCOMES},
                # 급등(+10% 터치)했으나 그 전에 손절된 비율 — 손절폭 진단용
                'stopped_before_hit': (round(float(stopped_winners.sum() / hits.sum()), 3)
                                       if hits.sum() else None),
            },
            'buckets': _buckets(done, h),
            # 게이트 판단용 — 종목별 비겹침 창 1건 (겹침 표본은 n을 부풀린다)
            'n_independent': len(indep),
            'buckets_independent': _buckets(indep, h),
        }

    if not stats['horizons']:
        if verbose:
            print('  [추적] 아직 평가 가능한 예측 없음 (기한 미도래)')
        return None

    stats['recent'] = _recent(records)

    # 구형 소비자(요약 스크립트 등) 호환: 주 기한 통계를 톱레벨에도 유지
    prim = stats['horizons'].get(str(SURGE_HORIZON))
    stats['n_evaluated']     = prim['n_evaluated'] if prim else 0
    stats['overall_hit_rate'] = prim['hit_rate'] if prim else None
    stats['buckets']          = prim['buckets'] if prim else []

    with open(STATS_PATH, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if verbose:
        for h in HOLD_HORIZONS:
            hs = stats['horizons'].get(str(h))
            if not hs:
                print(f'  [추적] {h}일 보유: 평가 가능 표본 없음 (기한 미도래)')
                continue
            aer = hs['avg_end_return']
            add = hs['avg_max_dd']
            extra = ''
            if aer is not None and add is not None:
                extra = (f'  보유종료수익 평균 {aer*100:+.1f}%'
                         f'  최대낙폭 평균 {add*100:.1f}%')
            print(f'  [추적] {h}일 보유({hs["n_evaluated"]}건) — '
                  f'+{SURGE_TARGET*100:.0f}% 터치율 {hs["hit_rate"]:.0%}{extra}')
            ru = hs['rule']
            sbh = (f'  급등군 중 손절 선행 {ru["stopped_before_hit"]:.0%}'
                   if ru['stopped_before_hit'] is not None else '')
            print(f'          규칙 승률 {ru["win_rate"]:.0%}  평균 {ru["avg_return"]*100:+.1f}%'
                  f'  [목표 {ru["target_rate"]:.0%} / 손절 {ru["stop_rate"]:.0%}'
                  f' / 기한 {ru["time_rate"]:.0%}]{sbh}')
            ind = {b['prob_min']: b for b in hs['buckets_independent']}
            for b in hs['buckets']:
                ib = ind.get(b['prob_min'])
                ind_s = f'  독립 {ib["hit_rate"]:.0%} ({ib["n"]}건)' if ib else ''
                print(f'          확률 {b["prob_min"]:.2f}~{b["prob_max"]:.2f}: '
                      f'{b["hit_rate"]:.0%} ({b["n"]}건){ind_s}')
        rc = stats.get('recent')
        if rc:
            print(f'  [최근] {rc["n_scans"]}회 스캔({rc["from"]}~{rc["to"]}, {rc["n"]}건, '
                  f'{rc["horizon"]}일) — 터치율 {rc["hit_rate"]:.0%}  '
                  f'규칙 승률 {rc["rule_win_rate"]:.0%}  평균 {rc["rule_avg_return"]*100:+.1f}%')
    return stats


def _load_stats() -> dict | None:
    if not STATS_PATH.exists():
        return None
    try:
        return json.loads(STATS_PATH.read_text(encoding='utf-8'))
    except Exception:
        return None


def recent_performance() -> dict | None:
    """최근 RECENT_SCANS회 스캔 실측 (evaluate()가 저장한 값)."""
    stats = _load_stats()
    return (stats or {}).get('recent')


def recommended_min_prob(default: float) -> tuple[float, str]:
    """주 기한(SURGE_HORIZON)의 독립 표본에서, 게이트 이상 픽의 누적 정밀도가
    TARGET_PRECISION 이상이 되는 최소 확률 구간 하한을 권고.
    누적 표본 TRACKER_MIN_BUCKET_N 미만이면 모델 임계값 유지.
    (구간별이 아니라 누적 정밀도를 쓰므로 상위 구간이 나빠지면 게이트가
    내려가지 않는다 — 날마다 0.25↔0.35로 흔들리던 문제 방지)"""
    stats = _load_stats()
    if stats is None:
        return default, '실측 표본 없음/손상 → 모델 임계값 사용'

    hs = stats.get('horizons', {}).get(str(SURGE_HORIZON), stats)
    buckets = hs.get('buckets_independent') or hs.get('buckets', [])
    n_total = hs.get('n_independent', hs.get('n_evaluated', 0))

    best = None
    for b in sorted(buckets, key=lambda b: b['prob_min']):
        upper = [u for u in buckets if u['prob_min'] >= b['prob_min']]
        n = sum(u['n'] for u in upper)
        if n < TRACKER_MIN_BUCKET_N:
            continue
        prec = sum(u['n'] * u['hit_rate'] for u in upper) / n
        if prec >= TARGET_PRECISION:
            best = (b['prob_min'], prec, n)
            break
    if best is None:
        return default, (f'{SURGE_HORIZON}일 독립 표본 {n_total}건 중 목표 정밀도 '
                         f'{TARGET_PRECISION:.0%} 달성 구간 없음 → 모델 임계값 사용')
    pmin, prec, n = best
    return max(default, pmin), (
        f'실측({SURGE_HORIZON}일, 독립 {n_total}건) 기반 게이트: 확률 {pmin:.2f}+ '
        f'누적 적중률 {prec:.0%} ({n}건)')

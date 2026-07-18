"""
예측 추적기 — v2 (v1의 존재하지 않던 stage_tracker를 리포 내부 모듈로 구현)

역할:
  1. save_predictions(): 매 스캔의 픽(티커/종가/확률/점수)을 predictions/에 JSONL로 기록
  2. evaluate(): 기한(HORIZON 거래일)이 지난 예측에 대해 실제로 +10%에 도달했는지
     yfinance로 확인 → 실현 적중률 계산, tracker_stats.json 갱신
  3. recommended_min_prob(): 실현 정밀도가 목표에 도달하는 최소 확률 구간 반환
     (다음 스캔의 게이트로 사용 — v1은 정밀도를 출력만 하고 버렸다)

predictions/ 와 tracker_stats.json 은 git에 커밋된다 — 학습 이력이
리포지토리만으로 재현 가능해야 한다 (v1은 전부 .gitignore에 걸려 있었다).
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from config import SURGE_TARGET, SURGE_HORIZON, TARGET_PRECISION
from data_fetcher import get_ohlcv

HERE = Path(__file__).parent
PRED_DIR   = HERE / 'predictions'
STATS_PATH = HERE / 'tracker_stats.json'
PRED_DIR.mkdir(exist_ok=True)

PROB_BUCKETS = [0.0, 0.15, 0.25, 0.35, 0.50, 1.01]


def save_predictions(rows: list[dict], scan_date: str) -> int:
    """rows: [{ticker, close, prob, tech_score, combined}, ...]"""
    path = PRED_DIR / f'preds_{scan_date}.jsonl'
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            rec = {'scan_date': scan_date, 'evaluated': False, **r}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return len(rows)


def _load_all() -> list[dict]:
    out = []
    for p in sorted(PRED_DIR.glob('preds_*.jsonl')):
        with open(p, encoding='utf-8') as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out


def _write_back(records: list[dict]) -> None:
    by_date: dict[str, list[dict]] = {}
    for r in records:
        by_date.setdefault(r['scan_date'], []).append(r)
    for d, rows in by_date.items():
        with open(PRED_DIR / f'preds_{d}.jsonl', 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')


def evaluate(verbose: bool = True) -> dict | None:
    """기한이 지난 미평가 예측을 실측과 대조. 실현 통계 갱신."""
    records = _load_all()
    if not records:
        return None

    # HORIZON 거래일 ≈ 1.5배 달력일 + 여유
    cutoff = datetime.now() - timedelta(days=int(SURGE_HORIZON * 1.6) + 2)
    pending = [r for r in records
               if not r.get('evaluated')
               and datetime.strptime(r['scan_date'], '%Y-%m-%d') <= cutoff]

    for r in pending:
        df = get_ohlcv(r['ticker'], period='6mo')
        if df.empty:
            continue
        after = df[df.index > r['scan_date']].head(SURGE_HORIZON)
        if len(after) < SURGE_HORIZON:
            continue
        max_gain = float(after['High'].max() / r['close'] - 1)
        r['evaluated'] = True
        r['max_gain']  = round(max_gain, 4)
        r['hit']       = max_gain >= SURGE_TARGET

    _write_back(records)

    done = [r for r in records if r.get('evaluated')]
    if not done:
        if verbose:
            print('  [추적] 아직 평가 가능한 예측 없음 (기한 미도래)')
        return None

    stats = {'updated_at': datetime.now().isoformat(),
             'n_evaluated': len(done),
             'overall_hit_rate': round(float(np.mean([r['hit'] for r in done])), 3),
             'buckets': []}
    probs = np.array([r.get('prob') if r.get('prob') is not None else -1 for r in done])
    hits  = np.array([r['hit'] for r in done])
    for lo, hi in zip(PROB_BUCKETS[:-1], PROB_BUCKETS[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum():
            stats['buckets'].append({
                'prob_min': lo, 'prob_max': hi,
                'n': int(m.sum()), 'hit_rate': round(float(hits[m].mean()), 3),
            })

    with open(STATS_PATH, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f'  [추적] 평가 완료 {len(done)}건 — 실현 적중률(+{SURGE_TARGET*100:.0f}%/'
              f'{SURGE_HORIZON}일): {stats["overall_hit_rate"]:.0%}')
        for b in stats['buckets']:
            print(f'          확률 {b["prob_min"]:.2f}~{b["prob_max"]:.2f}: '
                  f'{b["hit_rate"]:.0%} ({b["n"]}건)')
    return stats


def recommended_min_prob(default: float) -> tuple[float, str]:
    """실현 정밀도가 TARGET_PRECISION 이상인 최소 확률 버킷 하한을 게이트로 권고.
    표본 20건 미만이면 모델 기본 임계값 유지."""
    if not STATS_PATH.exists():
        return default, '실측 표본 없음 → 모델 임계값 사용'
    try:
        stats = json.loads(STATS_PATH.read_text(encoding='utf-8'))
    except Exception:
        return default, '통계 파일 손상 → 모델 임계값 사용'

    eligible = [b for b in stats.get('buckets', [])
                if b['n'] >= 20 and b['hit_rate'] >= TARGET_PRECISION]
    if not eligible:
        return default, (f'실측 {stats.get("n_evaluated", 0)}건 중 목표 정밀도 '
                         f'{TARGET_PRECISION:.0%} 달성 구간 없음 → 모델 임계값 사용')
    best = min(eligible, key=lambda b: b['prob_min'])
    return max(default, best['prob_min']), (
        f'실측 기반 게이트: 확률 {best["prob_min"]:.2f}+ 구간 '
        f'적중률 {best["hit_rate"]:.0%} ({best["n"]}건)')

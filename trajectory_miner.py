#!/usr/bin/env python3
"""
급등 궤적 분석 — 전조 패턴 매치 이후 5/10거래일의 "매일매일" 경로 대조

질문: "전조 패턴(pattern_rules.json)에 해당한 종목 중 실제로 +10% 급등한
종목은, 신호일 이후 하루하루 어떤 공통된 모습을 보이는가?"

방법 (look-ahead 없음 — 신호일 D0 기준 이후 실측만 사용):
  1. 3년치 유니버스에서 전조 규칙에 매치된 모든 신호일 D0 수집.
  2. D0 이후 D+1..D+10 창에서 일자별로 기록:
       누적 종가수익률 / 고가·저가 수익률 / 거래량비
  3. 급등군(10일 내 고가 +10% 터치) vs 비급등군으로 대조:
       - 첫 +10% 터치 시점 분포 (며칠째에 급등이 나오는가)
       - 일자별 누적수익 중앙값 경로 (급등군은 언제부터 벌어지는가)
       - 일자별 거래량비 중앙값 경로
       - 급등군이 터치 "이전"에 견딘 최대 낙폭 (인내 구간의 크기)
       - 조기 판별 체크포인트: D+2/D+3 수익률 구간별 최종 적중률
         (며칠째 어디에 있으면 계속 들고 가고, 어디면 자르는가)

결과: reports/trajectory_report.txt (한국어)

사용: python3 trajectory_miner.py
"""

import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from config import SURGE_TARGET, HISTORY_PERIOD, MIN_PRICE, MIN_AVG_VOLUME
from indicators import add_all
from surge_model import add_trend_deltas
from pattern_miner import load_rules
from data_fetcher import get_ohlcv, FetchLog
from universe import get_ticker_list

HERE = Path(__file__).parent
REPORTS_DIR = HERE / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)

H = 10   # 추적 창 (거래일) — 5일 관점은 동일 데이터의 D+1..D+5 열


def rule_mask(df: pd.DataFrame, rule: dict) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for c in rule['conditions']:
        col = df[c['feature']]
        m &= (col <= c['value']) if c['op'] == '<=' else (col > c['value'])
    return m.fillna(False)


def collect(rules: list[dict]) -> dict:
    tickers = get_ticker_list('TRAIN')
    log = FetchLog()
    rows = []          # 신호 1건 = dict
    for i, t in enumerate(tickers, 1):
        raw = get_ohlcv(t, period=HISTORY_PERIOD, log=log)
        if len(raw) < 260:
            continue
        df = add_all(raw)
        df = add_trend_deltas(df)
        base_ok = (df['Close'] >= MIN_PRICE) & (df['VolMA'] >= MIN_AVG_VOLUME)

        masks = {r['name']: (rule_mask(df, r) & base_ok) for r in rules}
        any_mask = np.logical_or.reduce([m.values for m in masks.values()])

        close = df['Close'].values
        high, low = df['High'].values, df['Low'].values
        vr = df['VolRatio'].values
        n = len(df)

        for idx in np.where(any_mask)[0]:
            if idx + H >= n:                       # 창이 덜 찬 신호는 제외
                continue
            c0 = close[idx]
            sl = slice(idx + 1, idx + 1 + H)
            cum_c = close[sl] / c0 - 1
            cum_h = high[sl] / c0 - 1
            cum_l = low[sl] / c0 - 1
            touch = np.where(cum_h >= SURGE_TARGET)[0]
            first_touch = int(touch[0]) + 1 if len(touch) else None   # D+k
            rows.append({
                'ticker': t,
                'rules': [name for name, m in masks.items() if m.values[idx]],
                'cum_close': cum_c, 'cum_low': cum_l,
                'vol_ratio': vr[sl],
                'hit10': first_touch is not None,
                'hit5': first_touch is not None and first_touch <= 5,
                'first_touch': first_touch,
                # 터치 이전(급등군) 또는 전체 창(비급등군)의 최저 저가
                'dd_before': float(cum_l[:first_touch].min()) if first_touch
                             else float(cum_l.min()),
            })
        if i % 20 == 0 or i == len(tickers):
            print(f'  [{i:3d}/{len(tickers)}] signals={len(rows)}', end='\r')
    print()
    for line in log.summary_lines():
        print(line)
    return rows


# ── 집계 ─────────────────────────────────────────────────────────────────────

def _median_path(rows: list[dict], key: str) -> np.ndarray:
    return np.nanmedian(np.vstack([r[key] for r in rows]), axis=0)


def build_report(rows: list[dict]) -> str:
    hits   = [r for r in rows if r['hit10']]
    misses = [r for r in rows if not r['hit10']]
    n, nh = len(rows), len(hits)
    sep, sep2 = '═' * 70, '─' * 70

    L = [sep,
         f'  급등 궤적 리포트 — 전조 패턴 매치 후 {H}거래일 일자별 경로',
         f'  신호 {n:,}건 (3y, 전조 규칙 매치)  |  10일 내 +10% 터치 {nh:,}건'
         f' ({nh/n:.1%})  |  5일 내 {sum(r["hit5"] for r in rows)/n:.1%}',
         f'  생성 {datetime.now().strftime("%Y-%m-%d %H:%M")}',
         sep, '']

    # 1. 첫 터치 시점 분포
    L.append('  [1] 급등군의 첫 +10% 터치 시점 (누적 비율)')
    L.append(sep2)
    ft = np.array([r['first_touch'] for r in hits])
    for d in range(1, H + 1):
        cum = (ft <= d).mean()
        bar = '█' * int(cum * 40)
        L.append(f'    D+{d:<2}까지 {cum:>5.1%}  {bar}')
    L.append('')

    # 2. 일자별 누적 종가수익 중앙값 경로
    L.append('  [2] 일자별 누적 종가수익률 중앙값 — 급등군 vs 비급등군')
    L.append(sep2)
    ph, pm = _median_path(hits, 'cum_close'), _median_path(misses, 'cum_close')
    L.append(f'    {"":>6} {"급등군":>8} {"비급등군":>8}   차이')
    for d in range(H):
        L.append(f'    D+{d+1:<3} {ph[d]:>7.1%} {pm[d]:>8.1%}   {ph[d]-pm[d]:>+6.1%}')
    L.append('')

    # 3. 일자별 거래량비 중앙값
    L.append('  [3] 일자별 거래량비(30일 평균 대비) 중앙값')
    L.append(sep2)
    vh, vm = _median_path(hits, 'vol_ratio'), _median_path(misses, 'vol_ratio')
    for d in range(H):
        L.append(f'    D+{d+1:<3} 급등군 {vh[d]:>5.2f}x   비급등군 {vm[d]:>5.2f}x')
    L.append('')

    # 4. 급등군이 터치 전에 견딘 낙폭
    L.append('  [4] 급등군이 +10% 터치 "이전"에 견딘 최대 낙폭 (저가 기준)')
    L.append(sep2)
    dd = np.array([r['dd_before'] for r in hits])
    for q, lab in [(25, '상위 25%'), (50, '중앙값'), (75, '하위 25%'), (90, '하위 10%')]:
        L.append(f'    {lab:<8} {np.percentile(dd, q):>7.1%}')
    ddm = np.array([r['dd_before'] for r in misses])
    L.append(f'    (비교) 비급등군 창 내 최대 낙폭 중앙값 {np.median(ddm):.1%}')
    L.append('')

    # 5. 조기 판별 체크포인트
    L.append('  [5] 조기 판별 체크포인트 — "D+k에 여기 있으면 최종 적중률은?"')
    L.append(sep2)
    edges = [-np.inf, -0.05, -0.02, 0.0, 0.02, 0.05, np.inf]
    labels = ['-5%↓', '-5~-2%', '-2~0%', '0~+2%', '+2~+5%', '+5%↑']
    for day in (2, 3, 5):
        c = np.array([r['cum_close'][day - 1] for r in rows])
        h = np.array([r['hit10'] for r in rows])
        L.append(f'    D+{day} 종가 기준:')
        for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
            m = (c > lo) & (c <= hi)
            if m.sum() >= 100:
                L.append(f'      {lab:>7}  →  최종 적중률 {h[m].mean():>5.1%}'
                         f'  (n={int(m.sum()):,})')
        L.append('')

    # 6. 규칙별
    L.append('  [6] 규칙별 성적 (신호 수 / 10일 내 터치율 / 터치 중앙 시점)')
    L.append(sep2)
    by_rule: dict[str, list[dict]] = {}
    for r in rows:
        for name in r['rules']:
            by_rule.setdefault(name, []).append(r)
    for name in sorted(by_rule, key=lambda k: -len(by_rule[k])):
        rs = by_rule[name]
        hr = np.mean([x['hit10'] for x in rs])
        fts = [x['first_touch'] for x in rs if x['first_touch']]
        med_ft = int(np.median(fts)) if fts else '-'
        L.append(f'    {name:<9} n={len(rs):>6,}  터치율 {hr:>5.1%}'
                 f'  중앙 터치 시점 D+{med_ft}')

    L += ['', sep,
          '  ※ 신호일이 연속되면 창이 겹칠 수 있음(같은 급등을 여러 신호가',
          '     공유). 절대 수치보다 급등군 vs 비급등군의 "차이"로 해석할 것.',
          sep]
    return '\n'.join(L)


def main():
    rules = load_rules()
    if not rules:
        raise SystemExit('pattern_rules.json 없음 — 먼저 pattern_miner.py 실행')
    print(f'\n  급등 궤적 분석 — 규칙 {len(rules)}개 매치 신호의 D+1..D+{H} 추적')
    rows = collect(rules)
    if not rows:
        raise SystemExit('매치 신호 없음')
    report = build_report(rows)
    path = REPORTS_DIR / 'trajectory_report.txt'
    path.write_text(report, encoding='utf-8')
    print(report)
    print(f'\n  리포트 저장: {path}')


if __name__ == '__main__':
    main()

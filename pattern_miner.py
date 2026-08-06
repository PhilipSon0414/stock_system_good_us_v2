#!/usr/bin/env python3
"""
급등 전조 패턴 마이너 — 확정된 투자 방식의 1단계 모듈

질문: "+10% 급등이 나오기 1주(5거래일) / 2주(10거래일) 전, 그 종목은
이평선·거래량 등에서 어떤 모양을 보이고 있었는가?"

방법 (look-ahead 없음):
  1. 학습 유니버스 전체(3y)의 매 거래일 D0에 대해
     라벨 = D0 종가 대비 이후 H거래일 내 고가가 +10% 도달 여부.
  2. D0 시점 지표(이평선 배열, 거래량비, OBV, 수축/확장, RSI 등)와
     직전 1주간 변화량(d*_5d)을 급등군 vs 비급등군으로 대조:
       - 이진 신호: 신호 발생 시 vs 미발생 시 적중률과 lift
       - 연속 지표: 5분위 구간별 적중률과 lift
  3. 얕은 결정트리(depth 3)로 "조건 조합 규칙"을 추출하고, 시간순
     홀드아웃(뒤 30%)에서 정밀도를 재검증해 과적합 규칙을 걸러낸다.
  4. 결과물:
       reports/pattern_report_h{H}.txt  — 사람이 읽는 패턴 리포트 (한국어)
       pattern_rules.json               — 검증 통과 규칙 (daily_scan이 사용)

사용:
  python3 pattern_miner.py            # H=5, 10 모두 분석 + 규칙 저장
  python3 pattern_miner.py 5          # 특정 기한만
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from config import (SURGE_TARGET, HISTORY_PERIOD, MIN_PRICE, MIN_AVG_VOLUME)
from indicators import add_all, MODEL_FEATURES
from surge_model import add_trend_deltas
from data_fetcher import get_ohlcv, FetchLog
from universe import get_ticker_list

HERE = Path(__file__).parent
REPORTS_DIR = HERE / 'reports'
RULES_PATH  = HERE / 'pattern_rules.json'
REPORTS_DIR.mkdir(exist_ok=True)

PATTERN_HORIZONS = [5, 10]   # 1주 / 2주

# ── 피처 사전 (리포트 한국어 표기 + 퍼센트 여부) ─────────────────────────────
BINARY_FEATURES = {
    'MA20AboveMA50':  '단기 정배열 (MA20>MA50)',
    'MA50AboveMA200': '중기 정배열 (MA50>MA200)',
    'GoldenCross':    'MA20/50 골든크로스 당일',
    'VCP':            'VCP (변동성 수축 패턴)',
    'NR7':            'NR7 (7일 최소 변동폭)',
    'VolRecovery':    '거래량 압축→회복 (3일+ 감소 후 반등)',
    'OBV_Diverge':    'OBV 다이버전스 (가격 횡보 + OBV 증가)',
    'RSI_Recovery':   'RSI 30~55 회복 구간',
    'NearBBLower':    '볼린저 하단 근접',
    'IsBull':         '당일 양봉',
}
CONT_FEATURES = {
    'VolRatio':      ('당일 거래량 / 30일 평균', 'x'),
    'VolSlope5':     ('거래량비 5일 기울기', ''),
    'VolDecDays':    ('연속 거래량 감소일 수', '일'),
    'OBVSlope5':     ('OBV 5일 기울기 (정규화)', ''),
    'PriceVsMA20':   ('종가의 MA20 대비 위치', '%'),
    'RSI14':         ('RSI(14)', ''),
    'StochK14':      ('스토캐스틱 %K(14)', ''),
    'ATRCompress':   ('ATR 압축도 (20일 평균 대비)', 'x'),
    'BBCompress':    ('볼린저밴드 폭 압축도', 'x'),
    'Dist52WPct':    ('52주 신고가와의 거리', '%'),
    'Gain5D':        ('최근 5일 수익률', '%'),
    'Gain10D':       ('최근 10일 수익률', '%'),
    'dVolRatio_5d':  ('거래량비 1주 변화', ''),
    'dRSI14_5d':     ('RSI 1주 변화', ''),
    'dATRCompress_5d': ('ATR 압축도 1주 변화', ''),
    'dBBCompress_5d':  ('BB 압축도 1주 변화', ''),
}
PCT_FEATURES = {'PriceVsMA20', 'Dist52WPct', 'Gain5D', 'Gain10D'}

TREE_FEATURES = list(CONT_FEATURES) + list(BINARY_FEATURES)


def _fmt(feat: str, v: float) -> str:
    if feat in PCT_FEATURES:
        return f'{v*100:+.1f}%'
    return f'{v:.2f}'


# ── 데이터 구축 ──────────────────────────────────────────────────────────────

def build_pattern_dataset(horizons: list[int], verbose: bool = True) -> pd.DataFrame:
    tickers = get_ticker_list('TRAIN')
    log = FetchLog()
    frames = []
    for i, t in enumerate(tickers, 1):
        raw = get_ohlcv(t, period=HISTORY_PERIOD, log=log)
        if len(raw) < 260:
            continue
        df = add_all(raw)
        df = add_trend_deltas(df)
        for h in horizons:
            fut_high = df['High'].rolling(h).max().shift(-h)
            gain = fut_high / df['Close'] - 1
            lab = (gain >= SURGE_TARGET).astype(float)
            lab[gain.isna()] = np.nan
            df[f'label_h{h}'] = lab
        mask = (df['Close'] >= MIN_PRICE) & (df['VolMA'] >= MIN_AVG_VOLUME)
        df = df[mask]
        df = df[df[[f'label_h{h}' for h in horizons]].notna().any(axis=1)]
        if len(df):
            cols = TREE_FEATURES + [f'label_h{h}' for h in horizons]
            keep = df[cols].copy()
            keep['ticker'] = t
            keep['date'] = df.index
            frames.append(keep)
        if verbose and (i % 20 == 0 or i == len(tickers)):
            print(f'  [{i:3d}/{len(tickers)}] rows={sum(len(f) for f in frames)}',
                  end='\r')
    if verbose:
        print()
        for line in log.summary_lines():
            print(line)
    if not frames:
        raise RuntimeError('패턴 데이터 구축 실패')
    return pd.concat(frames, ignore_index=True).sort_values('date').reset_index(drop=True)


# ── 단변량 분석 ──────────────────────────────────────────────────────────────

def binary_stats(d: pd.DataFrame, label: str) -> list[dict]:
    base = d[label].mean()
    out = []
    for f, name in BINARY_FEATURES.items():
        on = d[d[f] == 1.0][label]
        off = d[d[f] != 1.0][label]
        if len(on) < 200:
            continue
        out.append({'feature': f, 'name': name, 'n_on': len(on),
                    'hit_on': on.mean(), 'hit_off': off.mean(),
                    'lift': on.mean() / base if base else float('nan')})
    return sorted(out, key=lambda x: x['lift'], reverse=True)


def quintile_stats(d: pd.DataFrame, label: str) -> list[dict]:
    base = d[label].mean()
    out = []
    for f, (name, unit) in CONT_FEATURES.items():
        s = d[[f, label]].dropna()
        if len(s) < 1000:
            continue
        try:
            bins = pd.qcut(s[f], 5, duplicates='drop')
        except ValueError:
            continue
        g = s.groupby(bins, observed=True)[label].agg(['mean', 'count'])
        rows = [{'range': (iv.left, iv.right), 'hit': m, 'n': int(n),
                 'lift': m / base if base else float('nan')}
                for iv, m, n in zip(g.index, g['mean'], g['count'])]
        best = max(rows, key=lambda r: r['lift'])
        out.append({'feature': f, 'name': name, 'unit': unit,
                    'rows': rows, 'best_lift': best['lift']})
    return sorted(out, key=lambda x: x['best_lift'], reverse=True)


# ── 규칙 추출 (결정트리 + 시간순 홀드아웃 검증) ──────────────────────────────

def extract_rules(d: pd.DataFrame, label: str, horizon: int) -> list[dict]:
    d = d.dropna(subset=TREE_FEATURES + [label])
    split = int(len(d) * 0.7)          # 시간순 정렬 상태 유지
    train, hold = d.iloc[:split], d.iloc[split:]
    base_hold = hold[label].mean()

    tree = DecisionTreeClassifier(max_depth=3,
                                  min_samples_leaf=max(500, len(train) // 100),
                                  random_state=42)
    tree.fit(train[TREE_FEATURES].values, train[label].astype(int).values)

    t = tree.tree_
    rules = []

    def walk(node, conds):
        if t.children_left[node] == -1:           # leaf
            rules.append(list(conds))
            return
        f = TREE_FEATURES[t.feature[node]]
        thr = float(t.threshold[node])
        walk(t.children_left[node],  conds + [(f, '<=', thr)])
        walk(t.children_right[node], conds + [(f, '>', thr)])

    walk(0, [])

    # 같은 피처의 중복 조건 병합 → 홀드아웃 재검증
    out = []
    for conds in rules:
        merged: dict[tuple, float] = {}
        for f, op, thr in conds:
            key = (f, op)
            if op == '<=':
                merged[key] = min(merged.get(key, np.inf), thr)
            else:
                merged[key] = max(merged.get(key, -np.inf), thr)
        mask_h = pd.Series(True, index=hold.index)
        mask_t = pd.Series(True, index=train.index)
        for (f, op), thr in merged.items():
            if op == '<=':
                mask_h &= hold[f] <= thr
                mask_t &= train[f] <= thr
            else:
                mask_h &= hold[f] > thr
                mask_t &= train[f] > thr
        n_h = int(mask_h.sum())
        if n_h < 300:
            continue
        prec_h = float(hold.loc[mask_h, label].mean())
        lift_h = prec_h / base_hold if base_hold else float('nan')
        if lift_h < 1.25:                          # 홀드아웃에서도 유효한 규칙만
            continue
        out.append({
            'horizon': horizon,
            'conditions': [{'feature': f, 'op': op, 'value': round(thr, 6)}
                           for (f, op), thr in merged.items()],
            'precision_holdout': round(prec_h, 3),
            'lift_holdout': round(lift_h, 2),
            'n_holdout': n_h,
            'precision_train': round(float(train.loc[mask_t, label].mean()), 3),
            'base_holdout': round(float(base_hold), 3),
        })
    out.sort(key=lambda r: r['precision_holdout'], reverse=True)
    for i, r in enumerate(out, 1):
        r['name'] = f'H{horizon}-R{i}'
    return out[:6]


def describe_rule(rule: dict) -> str:
    parts = []
    for c in rule['conditions']:
        f, op, v = c['feature'], c['op'], c['value']
        if f in BINARY_FEATURES:
            name = BINARY_FEATURES[f]
            parts.append(name if (op == '>' and v < 1.0) else f'{name} 아님')
        else:
            name = CONT_FEATURES[f][0]
            parts.append(f'{name} {"≤" if op == "<=" else ">"} {_fmt(f, v)}')
    return ' + '.join(parts)


# ── 규칙 적용 (daily_scan 연동) ──────────────────────────────────────────────

def load_rules() -> list[dict]:
    if not RULES_PATH.exists():
        return []
    try:
        return json.loads(RULES_PATH.read_text(encoding='utf-8')).get('rules', [])
    except Exception:
        return []


def match_rules(df: pd.DataFrame, rules: list[dict] | None = None) -> list[dict]:
    """add_all() 적용된 df의 마지막 행이 만족하는 전조 규칙 목록.
    (트렌드 델타 컬럼이 없으면 여기서 추가)"""
    rules = rules if rules is not None else load_rules()
    if not rules:
        return []
    if 'dVolRatio_5d' not in df.columns:
        df = add_trend_deltas(df.copy())
    last = df.iloc[-1]
    matched = []
    for r in rules:
        ok = True
        for c in r['conditions']:
            v = last.get(c['feature'])
            try:
                v = float(v)
            except (TypeError, ValueError):
                ok = False
                break
            if np.isnan(v) or not (v <= c['value'] if c['op'] == '<=' else v > c['value']):
                ok = False
                break
        if ok:
            matched.append(r)
    return matched


# ── 리포트 ───────────────────────────────────────────────────────────────────

def build_report(d: pd.DataFrame, horizon: int) -> tuple[str, list[dict]]:
    label = f'label_h{horizon}'
    dd = d.dropna(subset=[label])
    base = dd[label].mean()
    week = '1주' if horizon == 5 else ('2주' if horizon == 10 else f'{horizon}일')
    sep, sep2 = '═' * 70, '─' * 70

    L = [sep,
         f'  급등 전조 패턴 리포트 — {week}({horizon}거래일) 내 +{SURGE_TARGET*100:.0f}% 도달',
         f'  표본 {len(dd):,}건 (유니버스 3년 전 거래일)  |  base rate {base:.1%}',
         f'  생성 {datetime.now().strftime("%Y-%m-%d %H:%M")}',
         sep,
         '',
         f'  해석: lift = 해당 상태일 때 적중률 / base rate({base:.1%}).',
         '  lift 1.3x 이상이면 유의미한 전조로 본다. 생존 편향이 있는',
         '  유니버스이므로 절대 수치보다 상대 비교(lift)로 해석할 것.',
         '']

    L.append(f'  [1] 이진 신호별 적중률 (신호 발생일 기준, {week} 내 +10% 도달률)')
    L.append(sep2)
    L.append(f'  {"신호":<38} {"발생시":>7} {"미발생":>7} {"lift":>6} {"표본":>9}')
    for b in binary_stats(dd, label):
        L.append(f'  {b["name"]:<38} {b["hit_on"]:>6.1%} {b["hit_off"]:>6.1%}'
                 f' {b["lift"]:>5.2f}x {b["n_on"]:>8,}')
    L.append('')

    L.append(f'  [2] 연속 지표 5분위 적중률 (최고 lift 구간 순)')
    L.append(sep2)
    for q in quintile_stats(dd, label):
        L.append(f'  ▸ {q["name"]} [{q["feature"]}]')
        for r in q['rows']:
            lo, hi = r['range']
            mark = ' ◀' if r['lift'] == q['best_lift'] else ''
            L.append(f'      {_fmt(q["feature"], lo):>8} ~ {_fmt(q["feature"], hi):>8}'
                     f'  적중 {r["hit"]:>6.1%}  lift {r["lift"]:.2f}x'
                     f'  (n={r["n"]:,}){mark}')
    L.append('')

    rules = extract_rules(d, label, horizon)
    L.append(f'  [3] 조건 조합 규칙 (결정트리 추출, 시간순 홀드아웃 30%로 재검증)')
    L.append(sep2)
    if rules:
        for r in rules:
            L.append(f'  {r["name"]}: {describe_rule(r)}')
            L.append(f'        홀드아웃 적중률 {r["precision_holdout"]:.1%}'
                     f'  (base {r["base_holdout"]:.1%}, lift {r["lift_holdout"]:.2f}x,'
                     f' n={r["n_holdout"]:,})')
    else:
        L.append('  홀드아웃 검증을 통과한 규칙 없음 (lift 1.25x 미만)')
    L += ['', sep,
          '  ※ 참고용 분석. 급등의 "전조"는 확률적 경향이며 보장이 아님.',
          sep]
    return '\n'.join(L), rules


def main():
    horizons = [int(a) for a in sys.argv[1:]] or PATTERN_HORIZONS
    print(f'\n  급등 전조 패턴 마이너 — 기한 {horizons} 거래일')
    print('  [1] 데이터 구축 (3y, TRAIN 유니버스)...')
    d = build_pattern_dataset(horizons)
    print(f'  전체 {len(d):,}건')

    all_rules = []
    for h in horizons:
        print(f'\n  [2] {h}거래일 패턴 분석...')
        report, rules = build_report(d, h)
        path = REPORTS_DIR / f'pattern_report_h{h}.txt'
        path.write_text(report, encoding='utf-8')
        print(report)
        print(f'\n  리포트 저장: {path}')
        all_rules += rules

    RULES_PATH.write_text(json.dumps({
        'generated_at': datetime.now().isoformat(),
        'target': SURGE_TARGET,
        'rules': all_rules,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n  규칙 저장: {RULES_PATH} ({len(all_rules)}개 — daily_scan이 자동 사용)')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
미국 주식 +10% 급등 후보 스캔 — v2

매일 장 마감 후 실행:
  1. 유니버스 스캔 → 기술 점수 (1차 게이트)
  2. 통과 종목만 재무/공매도 수집 + 급등 확률 모델 적용 (API 호출 최소화)
  3. 합산 점수 순위 + OB 매매 파라미터 + 리포트/이메일
  4. 예측을 tracker에 기록 → 기한 도래분 실측 평가 → 실현 적중률 기반 게이트 갱신

v1 대비 변경점:
  - 존재하지 않는 모듈(surge_predictor, stage_tracker) 의존 제거 — 전부 리포 내부
  - 모델 없으면 합산에 0을 조용히 섞지 않고 리포트에 '모델 미학습' 명시
  - 데이터 수집 실패 목록을 리포트에 출력 (조용한 skip 금지)
  - SPY는 스캔당 1회만 수집 (v1은 종목마다 재수집)
  - 코드 자동 git push 제거 (검토 없는 자동 커밋 위험)
  - price_vs_ma20 == 0.0 이 ❌로 표시되던 falsy 버그 수정

사용: python3 daily_scan.py [ALL|SP500|GROWTH] [YYYY-MM-DD]
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from config import (MIN_PRICE, MIN_AVG_VOLUME, TECH_SCORE_GATE, FINAL_MIN_SCORE,
                    TOP_N_REPORT, TOP_N_DETAIL, W_SCORE, W_MODEL,
                    SURGE_TARGET, SURGE_HORIZON)
from universe import get_ticker_list
from data_fetcher import (get_ohlcv, get_info, short_signal, fmt_market_cap,
                          FetchLog)
from indicators import add_all
from order_block import get_order_blocks, calc_trade_params
from scorer import score_technical
from surge_model import load_model, predict_prob
from email_sender import send_report
import tracker

HERE = Path(__file__).parent
REPORTS_DIR = HERE / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)


# ── 단일 종목 분석 ───────────────────────────────────────────────────────────

def analyze_phase1(ticker: str, end_date: str | None, log: FetchLog) -> dict | None:
    """1차: 가격/유동성 필터 + 기술 점수. info/모델은 아직 안 씀."""
    df = get_ohlcv(ticker, period='2y', end_date=end_date, log=log)
    if len(df) < 260:
        return None
    close = float(df['Close'].iloc[-1])
    if close < MIN_PRICE:
        return None

    df = add_all(df)
    avg_vol = float(df['VolMA'].iloc[-1]) if df['VolMA'].notna().iloc[-1] else 0
    if avg_vol < MIN_AVG_VOLUME:
        return None

    ob = get_order_blocks(df)
    tech, tags = score_technical(df, ob, info=None)
    if tech < TECH_SCORE_GATE:
        return None

    latest = df.iloc[-1]
    pvm = latest.get('PriceVsMA20')
    return {
        'ticker': ticker, 'price': close, 'df': df, 'ob': ob,
        'tech_score': tech, 'tags': tags,
        'vol_ratio': float(latest.get('VolRatio') or 0),
        'price_vs_ma20': round(float(pvm) * 100, 1) if pvm is not None and pvm == pvm else None,
        'gain5d': float(latest.get('Gain5D') or 0),
    }


def analyze_phase2(r: dict, model_payload: dict | None, spy_ret20: float | None,
                   log: FetchLog) -> dict:
    """2차: 재무/공매도 + 모델 확률 + 최종 점수."""
    info = get_info(r['ticker'], log=log)
    tech, tags = score_technical(r['df'], r['ob'], info=info)
    r['tech_score'], r['tags'] = tech, tags
    r['info'] = info
    r['name'] = info.get('name', r['ticker'])
    r['sector'] = info.get('sector', '')

    # 급등 확률 (보정된 값). 모델 없으면 None — 조용히 0 섞지 않음.
    pred = None
    if model_payload is not None:
        try:
            pred = predict_prob(r['df'], loaded=model_payload)
        except Exception as e:
            print(f'  ⚠ {r["ticker"]} 모델 예측 실패: {str(e)[:60]}')
    r['pred'] = pred

    if pred is not None:
        r['combined'] = round(W_SCORE * tech + W_MODEL * pred['prob'] * 100)
    else:
        r['combined'] = tech

    # RS vs SPY (20일)
    r['rs_vs_spy'] = None
    if spy_ret20 is not None and len(r['df']) >= 21:
        c = r['df']['Close']
        stock_ret = (float(c.iloc[-1]) / float(c.iloc[-21]) - 1) * 100
        r['rs_vs_spy'] = round(stock_ret - spy_ret20, 1)
    return r


# ── 엘리트픽 ─────────────────────────────────────────────────────────────────

def get_elite_picks(results: list, model_ok: bool) -> list:
    """확률 게이트 통과 + 과열 아님 + 기술점수 상위. 모델이 없으면
    기술점수 단독 기준이며 리포트에 그 사실이 표시된다."""
    elite = []
    for r in results:
        reasons = []
        pred = r.get('pred')
        overheated = r.get('gain5d', 0) >= 0.15
        if overheated:
            continue
        if model_ok and pred and pred['passes']:
            reasons.append(f'급등확률 {pred["prob"]:.0%} (임계값 {pred["threshold"]:.0%} 통과)')
        if r['tech_score'] >= 75:
            reasons.append(f'기술점수 {r["tech_score"]}')
        sp = (r.get('info') or {}).get('short_pct')
        if sp is not None and sp >= 20 and r['tech_score'] >= 60:
            reasons.append(f'공매도 {sp:.0f}% 스퀴즈 후보')
        if reasons:
            elite.append({**r, 'elite_reasons': reasons})
    return sorted(elite, key=lambda x: x['combined'], reverse=True)[:10]


# ── 리포트 ───────────────────────────────────────────────────────────────────

def build_report(results: list, elite: list, market: str, model_ok: bool,
                 gate_note: str, log: FetchLog, scan_date: str) -> str:
    sep, sep2 = '═' * 70, '─' * 70
    L = [sep,
         f'  미국 주식 +{SURGE_TARGET*100:.0f}% 급등 후보 스캔 (v2)',
         f'  기준일: {scan_date}  |  시장: {market}  |  후보 {len(results)}종목',
         sep]

    if model_ok:
        L.append(f'  합산 = 기술점수 {W_SCORE*100:.0f}% + 급등확률 {W_MODEL*100:.0f}%'
                 f'  (확률은 {SURGE_HORIZON}거래일 내 +{SURGE_TARGET*100:.0f}% 도달, 보정값)')
        L.append(f'  게이트: {gate_note}')
    else:
        L.append('  ⚠ 급등 확률 모델 미학습 — 합산은 기술점수 단독입니다.')
        L.append('    `python3 surge_model.py train` 실행 후 확률이 반영됩니다.')
    L += log.summary_lines()
    L.append('')

    # 엘리트픽
    L.append(sep)
    if elite:
        L.append('  ★★★ 엘리트픽 (확률 게이트 통과 + 과열 아님)')
        L.append(sep2)
        for r in elite:
            pred = r.get('pred')
            prob_s = f'{pred["prob"]:.0%}' if pred else '—'
            L.append(f'  {r["ticker"]:<7} {r["name"][:22]:<22} ${r["price"]:>9.2f}'
                     f'  합산 {r["combined"]:>3}  확률 {prob_s:>4}'
                     f'  | {" | ".join(r["elite_reasons"])}')
    else:
        L.append('  ★★★ 엘리트픽: 해당 없음')
    L.append(sep2)
    L.append('')

    # 전체 순위
    L.append(f'  {"순위":<4} {"티커":<7} {"이름":<22} {"가격":>9} {"합산":>4}'
             f' {"기술":>4} {"확률":>5} {"거래량":>6} {"vsMA20":>7}  섹터')
    L.append(sep2)
    for rank, r in enumerate(results, 1):
        pred = r.get('pred')
        prob_s = f'{pred["prob"]*100:.0f}%' if pred else '  —'
        pvm = r.get('price_vs_ma20')
        # v1 버그 수정: 0.0도 정상 표기 (falsy 체크 금지)
        pvm_s  = f'{pvm:+.1f}%' if pvm is not None else '   —'
        pvm_fl = ('✅' if pvm >= 0 else ('⚡' if pvm >= -5 else '❌')) if pvm is not None else ' '
        L.append(f'  {rank:<4} {r["ticker"]:<7} {r["name"][:22]:<22}'
                 f' ${r["price"]:>8.2f} {r["combined"]:>4} {r["tech_score"]:>4}'
                 f' {prob_s:>5} {r["vol_ratio"]:>5.1f}x {pvm_s:>6}{pvm_fl}  {r.get("sector","")[:12]}')
    L.append(sep2)
    L.append('')

    # 상세
    L.append(sep)
    L.append(f'  [ 상위 {min(TOP_N_DETAIL, len(results))}종목 상세 + OB 매매 파라미터 ]')
    L.append(sep)
    for rank, r in enumerate(results[:TOP_N_DETAIL], 1):
        pred = r.get('pred')
        info = r.get('info', {})
        L.append(f'\n  {rank:>2}. {r["ticker"]} — {r["name"]}')
        prob_line = (f'급등확률 {pred["prob"]:.0%}'
                     + (' (게이트 통과 ✅)' if pred['passes'] else ' (게이트 미달)')
                     if pred else '급등확률 — (모델 미학습)')
        L.append(f'      합산 {r["combined"]}  |  기술 {r["tech_score"]}  |  {prob_line}')
        rs = r.get('rs_vs_spy')
        rs_s = f'  RS(vs SPY) {rs:+.1f}%' if rs is not None else ''
        L.append(f'      ${r["price"]:.2f}  거래량 {r["vol_ratio"]:.1f}x{rs_s}'
                 f'  시총 {fmt_market_cap(info.get("market_cap", 0))}'
                 f'  섹터 {info.get("sector","")}')
        ss = short_signal(info)
        if ss:
            L.append(f'      {ss}')
        L.append(f'      신호: {" | ".join(r["tags"][:4])}')

        high52w = None
        if 'High52W' in r['df'].columns:
            v = r['df']['High52W'].iloc[-1]
            high52w = float(v) if v == v else None
        tp = calc_trade_params(r['ob'], r['price'], high52w, r['tech_score'])
        L.append(f'      진입 ${tp["entry"]:.2f} ({tp["entry_desc"]})')
        if tp.get('target'):
            L.append(f'      목표 ${tp["target"]:.2f} ({tp.get("expected_return","?")}%'
                     f', {tp["target_desc"]})')
        if tp.get('stop'):
            L.append(f'      손절 ${tp["stop"]:.2f} ({tp["stop_desc"]})'
                     + (f'  R:R {tp["rr_ratio"]:.1f}:1' if tp.get('rr_ratio') else ''))

    L += ['', sep,
          '  ※ 참고용 분석입니다. 투자 판단과 책임은 본인에게 있습니다.',
          '  ※ 백테스트는 생존 편향이 있는 유니버스 기준 — 실현 적중률은',
          '     tracker_stats.json (실측 기반)을 신뢰하세요.',
          sep]
    return '\n'.join(L)


# ── 메인 ─────────────────────────────────────────────────────────────────────

def get_spy_ret20(end_date: str | None, log: FetchLog) -> float | None:
    spy = get_ohlcv('SPY', period='1y', end_date=end_date, log=log)
    if len(spy) >= 21:
        return (float(spy['Close'].iloc[-1]) / float(spy['Close'].iloc[-21]) - 1) * 100
    return None


def run_scan(market: str = 'ALL', scan_date: str | None = None) -> list:
    log = FetchLog()
    tickers = get_ticker_list(market)
    print(f'\n  미국 주식 +{SURGE_TARGET*100:.0f}% 급등 후보 스캔 (v2)')
    print(f'  대상 {len(tickers)}종목  |  시장 {market}'
          f'  |  기준일 {scan_date or "최신"}')

    model_payload = load_model()
    if model_payload is None:
        print('  ⚠ 모델 미학습 — 기술점수 단독 모드 (surge_model.py train 권장)')

    # 1차: 기술 게이트
    phase1 = []
    for i, t in enumerate(tickers, 1):
        r = analyze_phase1(t, scan_date, log)
        if r:
            phase1.append(r)
        if i % 20 == 0 or i == len(tickers):
            print(f'  1차 스캔 {i}/{len(tickers)}  통과 {len(phase1)}', end='\r')
        time.sleep(0.05)
    print(f'\n  1차 통과: {len(phase1)}종목 (기술점수 {TECH_SCORE_GATE}+)')

    # 2차: info + 모델
    spy_ret20 = get_spy_ret20(scan_date, log)
    results = [analyze_phase2(r, model_payload, spy_ret20, log) for r in phase1]
    results = [r for r in results if r['combined'] >= FINAL_MIN_SCORE]
    results.sort(key=lambda x: x['combined'], reverse=True)
    results = results[:TOP_N_REPORT]

    return results, model_payload is not None, log


def main(market: str = 'ALL', scan_date: str | None = None):
    results, model_ok, log = run_scan(market, scan_date)
    date_str = scan_date or datetime.now().strftime('%Y-%m-%d')

    if not results:
        print('  발굴 종목 없음.')
        for line in log.summary_lines():
            print(line)
        return

    # 실측 기반 게이트
    default_thr = 0.0
    if model_ok:
        default_thr = load_model()['threshold']
    gate, gate_note = tracker.recommended_min_prob(default_thr)

    elite = get_elite_picks(results, model_ok)
    report = build_report(results, elite, market, model_ok, gate_note, log, date_str)
    print('\n' + report)

    # 저장 (텍스트 리포트 + 요약 이메일용 JSON)
    fname = f'{date_str}_scan'
    (REPORTS_DIR / f'{fname}.txt').write_text(report, encoding='utf-8')
    scan_json = {
        'scan_date': date_str, 'market': market, 'model_ok': model_ok,
        'gate_note': gate_note,
        'results': [{
            'ticker': r['ticker'], 'name': r['name'], 'price': r['price'],
            'combined': r['combined'], 'tech_score': r['tech_score'],
            'prob': (r.get('pred') or {}).get('prob'),
            'passes_gate': (r.get('pred') or {}).get('passes', False),
            'vol_ratio': r['vol_ratio'], 'price_vs_ma20': r.get('price_vs_ma20'),
            'rs_vs_spy': r.get('rs_vs_spy'), 'sector': r.get('sector', ''),
            'short_pct': (r.get('info') or {}).get('short_pct'),
            'elite': any(e['ticker'] == r['ticker'] for e in elite),
        } for r in results],
    }
    (REPORTS_DIR / f'{fname}.json').write_text(
        json.dumps(scan_json, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n  리포트 저장: reports/{fname}.txt / .json')

    # 예측 기록 + 실측 평가
    tracker.save_predictions([{
        'ticker': r['ticker'], 'close': r['price'],
        'prob': (r.get('pred') or {}).get('prob'),
        'tech_score': r['tech_score'], 'combined': r['combined'],
    } for r in results], scan_date=date_str)
    tracker.evaluate(verbose=True)

    # 이메일
    subject = (f'[미국주식 v2] {date_str} — 엘리트픽 {len(elite)}종목'
               if elite else f'[미국주식 v2] {date_str} — 후보 {len(results)}종목')
    send_report(subject=subject, body=report,
                attachment_path=str(REPORTS_DIR / f'{fname}.txt'))
    print(f'  완료. {len(results)}종목 발굴, 엘리트픽 {len(elite)}종목.')


if __name__ == '__main__':
    market_arg = sys.argv[1] if len(sys.argv) > 1 else 'ALL'
    date_arg   = sys.argv[2] if len(sys.argv) > 2 else None
    main(market_arg, scan_date=date_arg)

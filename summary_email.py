#!/usr/bin/env python3
"""
스캔 요약 이메일 — v2

v1 버그 수정: v1의 send_us_summary_email.py는 특정 날짜(2026-06-04)의 결과가
통째로 하드코딩되어 있어, 언제 실행해도 한 달 전 추천을 오늘 날짜로 발송했다.
v2는 daily_scan이 저장한 최신 reports/*_scan.json을 파싱해 요약을 생성한다.
하드코딩 없음.

사용: python3 summary_email.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from email_sender import send_report
from config import SURGE_TARGET, SURGE_HORIZON

REPORTS_DIR = HERE / 'reports'


def latest_scan() -> tuple[dict, Path] | None:
    files = sorted(REPORTS_DIR.glob('*_scan.json'), reverse=True)
    if not files:
        return None
    data = json.loads(files[0].read_text(encoding='utf-8'))
    txt = files[0].with_suffix('.txt')
    return data, (txt if txt.exists() else None)


def build_summary(scan: dict) -> str:
    sep, sep2 = '═' * 64, '─' * 64
    results = scan['results']
    elite   = [r for r in results if r.get('elite')]
    now     = datetime.now().strftime('%Y-%m-%d %H:%M')

    L = [sep,
         f'  미국 주식 +{SURGE_TARGET*100:.0f}% 급등 후보 — 요약',
         f'  스캔 기준일: {scan["scan_date"]}  |  발송: {now}  |  후보 {len(results)}종목',
         sep]
    if not scan.get('model_ok'):
        L.append('  ⚠ 확률 모델 미학습 — 기술점수 단독 순위입니다.')
    else:
        L.append(f'  게이트: {scan.get("gate_note", "")}')
    L.append('')

    if elite:
        L.append('★★★ 엘리트픽')
        L.append(sep2)
        for r in elite:
            prob = f'{r["prob"]*100:.0f}%' if r.get('prob') is not None else '—'
            L.append(f'  {r["ticker"]:<7} {r["name"][:20]:<20} ${r["price"]:>9.2f}'
                     f'  합산 {r["combined"]:>3}  확률 {prob:>4}')
        L.append(sep2)
        L.append('')

    L.append('전체 TOP 10')
    L.append(sep2)
    for i, r in enumerate(results[:10], 1):
        prob = f'{r["prob"]*100:.0f}%' if r.get('prob') is not None else '  —'
        pvm  = r.get('price_vs_ma20')
        pvm_s = f'MA20 {pvm:+.1f}%' if pvm is not None else ''
        L.append(f'  {i:>2}. {r["ticker"]:<7} {r["name"][:18]:<18} ${r["price"]:>9.2f}'
                 f'  합산 {r["combined"]:>3}  확률 {prob:>4}  {pvm_s}')
    L.append(sep2)
    L += ['',
          f'  ※ 확률 = {SURGE_HORIZON}거래일 내 +{SURGE_TARGET*100:.0f}% 도달 (보정값)',
          '  ※ 참고용. 투자 판단은 본인 책임.',
          sep]
    return '\n'.join(L)


def main():
    loaded = latest_scan()
    if not loaded:
        print('  reports/에 스캔 결과 JSON이 없습니다. daily_scan.py를 먼저 실행하세요.')
        return
    scan, txt_path = loaded
    body = build_summary(scan)
    n_elite = sum(1 for r in scan['results'] if r.get('elite'))
    subject = (f'[미국주식 v2 요약] {scan["scan_date"]} — '
               f'엘리트픽 {n_elite}종목 / 후보 {len(scan["results"])}종목')
    ok = send_report(subject=subject, body=body,
                     attachment_path=str(txt_path) if txt_path else None)
    print('  요약 이메일 발송 완료' if ok else '  이메일 발송 실패/비활성화')
    print(body)


if __name__ == '__main__':
    main()

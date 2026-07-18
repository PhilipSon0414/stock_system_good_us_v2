"""
Smart Money 기술 점수 (0~100) — v2

v1 대비 변경점:
  1. 섹션 상한을 min()으로 강제 — v1은 '모멘텀 최대 20점' 섹션이 실제로는
     79점까지 가능해 상위권 점수가 포화되어 변별력이 사라졌다.
  2. 모순 신호 이중 가점 제거 — 변동성 수축(VCP/NR7)과 확대(ATR/BB 확대)는
     상호 배타적 셋업이므로 두 경로 중 높은 쪽 하나만 반영.
  3. 자체 분석에서 변별력이 없었던 RSI 35~55 가점(D-1 평균 51.1 vs 대조군
     52.3) 제거. 과매수 페널티만 유지.
  4. 추격 매수 방지 강화 — 5일 +15% / 10일 +25% 이미 급등 시 강한 페널티.
     "급등 확인"이 아니라 "급등 이전"을 잡는 것이 목표.
  5. info 데이터 실패 시 0으로 조작하지 않음 — data_ok=False면 해당 항목 제외.

점수 구성 (상한 강제):
  추세          최대 20
  거래량/수급    최대 30
  오더블록      최대 20
  셋업(수축 또는 확장 중 하나)  최대 20
  쇼트스퀴즈    최대 10
  페널티는 합산 후 차감, 최종 0~100 클램프
"""

import numpy as np
import pandas as pd

from config import OVERHEAT_GAIN_5D, OVERHEAT_GAIN_10D


def _f(latest: pd.Series, col: str, default: float = np.nan) -> float:
    v = latest.get(col, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def score_technical(df: pd.DataFrame, ob_info: dict,
                    info: dict | None = None) -> tuple[int, list[str]]:
    """기술적 Smart Money 점수. info는 없어도 동작 (2단계 스캔용)."""
    if len(df) < 60:
        return 0, []

    latest = df.iloc[-1]
    tags: list[str] = []
    close = _f(latest, 'Close')

    # ── 1. 추세 (최대 20) ────────────────────────────────────────────────────
    trend = 0
    pvm20 = _f(latest, 'PriceVsMA20')
    ma_bull = bool(latest.get('MaBull', False))
    ma20_50 = _f(latest, 'MA20AboveMA50') == 1.0
    price_ok = not np.isnan(pvm20) and pvm20 >= -0.02

    if ma_bull and price_ok:
        trend += 20; tags.append('정배열 (MA20>50>200, 가격 위)')
    elif ma_bull:
        trend += 6;  tags.append('정배열이나 가격이 MA20 아래')
    elif ma20_50 and price_ok:
        trend += 12; tags.append('단기 정배열 (MA20>50)')
    trend = min(20, trend)

    # ── 2. 거래량/수급 (최대 30) ─────────────────────────────────────────────
    vol = 0
    vr        = _f(latest, 'VolRatio')
    vol_rec   = _f(latest, 'VolRecovery') == 1.0
    obv_div   = _f(latest, 'OBV_Diverge') == 1.0
    obv_slope = _f(latest, 'OBVSlope5')

    if not np.isnan(vr):
        if 1.5 <= vr < 3.0:
            vol += 12; tags.append(f'거래량 {vr:.1f}x 증가')
        elif 1.2 <= vr < 1.5:
            vol += 8;  tags.append(f'거래량 {vr:.1f}x 활성')
        elif vr >= 3.0:
            # 3x 이상 폭발은 이미 이벤트가 터진 경우가 많음 — 만점 주지 않음
            vol += 6;  tags.append(f'거래량 {vr:.1f}x 폭발 (이미 발동 가능성)')
    if vol_rec:
        vol += 10; tags.append('★ 거래량 압축→회복')
    elif not np.isnan(obv_slope) and obv_slope > 0 and not np.isnan(vr) and vr < 1.5:
        vol += 4; tags.append('거래량 증가 추세')
    if obv_div:
        vol += 10; tags.append('★ OBV 다이버전스 — 조용한 매집')
    if info and info.get('data_ok'):
        inst = info.get('inst_pct')
        if inst is not None and inst >= 80:
            vol += 4; tags.append(f'기관 {inst:.0f}% 보유')
    vol = min(30, vol)

    # ── 3. 오더블록 (최대 20) ────────────────────────────────────────────────
    obp = 0
    bull_dist = ob_info.get('nearest_bull_dist')
    if bull_dist is not None:
        if bull_dist == 0.0:
            obp += 14; tags.append('강세OB 진입 중')
        elif bull_dist < 0.015:
            obp += 12; tags.append(f'강세OB 근접 {bull_dist*100:.1f}%')
        elif bull_dist < 0.03:
            obp += 9
        elif bull_dist < 0.06:
            obp += 5
        if ob_info.get('bull_fresh') and bull_dist < 0.05:
            obp += 4; tags.append('미접촉 강세OB')
    flip_dist = ob_info.get('nearest_flipped_dist')
    if flip_dist is not None and flip_dist < 0.04:
        obp += 8; tags.append(f'OB 플립 지지 {flip_dist*100:.1f}%')
    obp = min(20, obp)

    # ── 4. 셋업 (최대 20) — 수축과 확장 중 하나만 ───────────────────────────
    # 수축형: 폭발 전 에너지 응축 (VCP/NR7 + 거래량 압축)
    contraction = 0
    if _f(latest, 'VCP') == 1.0:
        contraction += 12;
    if _f(latest, 'NR7') == 1.0:
        contraction += 4
    if not np.isnan(vr) and vr < 0.7:
        contraction += 4

    # 확장형: 이미 시동 건 모멘텀 (ATR/BB 확대 + 신고가 근접)
    expansion = 0
    atr_c = _f(latest, 'ATRCompress')
    bb_c  = _f(latest, 'BBCompress')
    d52   = _f(latest, 'Dist52WPct')
    if not np.isnan(atr_c) and atr_c >= 1.15:
        expansion += 6
    if not np.isnan(bb_c) and bb_c >= 1.15:
        expansion += 4
    if not np.isnan(d52):
        if d52 <= 0.02:
            expansion += 8
        elif d52 <= 0.05:
            expansion += 5
    if _f(latest, 'GoldenCross') == 1.0:
        expansion += 5

    setup = min(20, max(contraction, expansion))
    if setup > 0:
        if contraction >= expansion:
            tags.append(f'수축형 셋업 {min(20,contraction)}점 (VCP/NR7/거래량압축)')
        else:
            tags.append(f'확장형 셋업 {min(20,expansion)}점 (ATR/BB확대/신고가근접)')

    # ── 5. 쇼트스퀴즈 (최대 10) ──────────────────────────────────────────────
    squeeze = 0
    if info and info.get('data_ok'):
        sp = info.get('short_pct')
        if sp is not None:
            if sp >= 20:
                squeeze = 10; tags.append(f'공매도 {sp:.0f}% 쇼트스퀴즈 후보 (격주 데이터 주의)')
            elif sp >= 10:
                squeeze = 5;  tags.append(f'공매도 {sp:.0f}%')

    pts = trend + vol + obp + setup + squeeze

    # ── 페널티 ───────────────────────────────────────────────────────────────
    gain5  = _f(latest, 'Gain5D')
    gain10 = _f(latest, 'Gain10D')
    rsi    = _f(latest, 'RSI14')

    if not np.isnan(gain5) and gain5 >= OVERHEAT_GAIN_5D:
        pts -= 20; tags.append(f'⚠ 5일 +{gain5*100:.0f}% 이미 급등 — 추격 위험')
    if not np.isnan(gain10) and gain10 >= OVERHEAT_GAIN_10D:
        pts -= 20; tags.append(f'⚠ 10일 +{gain10*100:.0f}% 이미 급등')
    if not np.isnan(rsi) and rsi > 75:
        pts -= 8; tags.append(f'⚠ RSI {rsi:.0f} 과매수')
    if bool(latest.get('MaBear', False)):
        pts -= 12; tags.append('⚠ 역배열 (하락 추세)')
    if not np.isnan(pvm20) and pvm20 < -0.15:
        pts -= 12; tags.append(f'⚠ MA20 대비 {pvm20*100:.0f}% 이탈')

    return int(max(0, min(100, pts))), tags

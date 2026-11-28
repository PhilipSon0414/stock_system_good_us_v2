"""미너비니식 VCP(변동성 축소 패턴) 돌파 전략 — 신호 판정과 청산 시뮬레이션.

신호(모두 신호일 종가까지의 데이터만 사용, look-ahead 없음):
  1. 시장 필터   : SPY 종가가 50일·200일 이동평균선 위
  2. 추세 템플릿 : 종가 > MA50 > MA150 > MA200, MA200 상승, 52주 신고가 대비 -25% 이내
  3. 상대강도    : 유니버스 내 3/6/9/12개월 가중 수익률 백분위 ≥ RS_MIN
  4. VCP         : 베이스 안에서 조정 폭이 2회 이상 단계적으로 축소, 마지막 조정 ≤ 10%
  5. 거래량 고갈 : 돌파 직전 10일 중 가장 한산한 3일 평균 ≤ 50일 평균의 DRYUP_MAX
  6. 돌파        : 종가가 피벗(마지막 수축 구간 고점) 상향 돌파 + 거래량 ≥ 50일 평균 × 1.5,
                   피벗 대비 +5% 초과 종가는 추격 금지

진입·청산(일봉 기준, 장 마감 후 판정 → 다음 거래일 시가 진입):
  - 다음 시가가 피벗 +5% 초과면 진입 포기
  - 손절 -7% (진입가 기준, 갭 하락은 시가 체결)
  - +15% 도달 시 절반 익절, 잔여분 손절을 본전으로 상향
  - 잔여분은 종가가 20일선을 이탈하면 다음 거래일 시가 청산
  - 같은 날 손절·익절이 겹치면 손절 우선(보수적). 비용은 execution.policy와 동일.

실적(EPS·매출 성장) 조건은 과거 시점의 재무 데이터를 확보할 수 없어 백테스트에 넣지 않았다.
이 모듈은 연구/검증용이며 daily_scan의 추천·코칭 경로를 바꾸지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VERSION = 'vcp_v1'

PARAMS = {
    'rs_min': 0.80,            # 유니버스 내 RS 백분위 하한
    'max_dist_high': 0.25,     # 52주 신고가 대비 최대 이격
    'base_min': 15,            # 베이스 최소 길이(거래일)
    'base_max': 120,           # 베이스 탐색 범위(거래일)
    'swing_min': 0.03,         # 스윙 인식 최소 반전폭 (ATR이 크면 1.5×ATR)
    'first_depth_min': 0.08,   # 가장 깊은 조정 하한 — 조정 없는 직선 상승 제외
    'first_depth_max': 0.35,   # 가장 깊은 조정 상한 — 무너진 차트 제외
    'last_depth_max': 0.10,    # 마지막 조정 폭 상한
    'shrink': 0.85,            # 마지막 조정 ≤ 직전 조정 × shrink
    'dryup_max': 0.60,         # 거래량 고갈 기준 (50일 평균 대비)
    'breakout_vol': 1.5,       # 돌파일 거래량 / 50일 평균
    'max_chase': 0.05,         # 피벗 대비 추격 한도
    'stop_pct': 0.07,          # 초기 손절
    'partial_at': 0.15,        # 부분 익절 수익률
    'partial_frac': 0.5,       # 부분 익절 비중
    'trail_ma': 20,            # 트레일링 이동평균
    'trail_from_start': False, # True면 부분 익절 전에도 MA 이탈 시 청산
    'max_hold': 120,           # 안전장치: 최대 보유 거래일
    'fee_bps': 5.0, 'slippage_bps': 10.0,
}


def rs_percentile(closes: pd.DataFrame) -> pd.DataFrame:
    """종목×날짜 종가 패널 → 날짜별 유니버스 내 상대강도 백분위(0~1).
    IBD식 가중: 최근 3개월 40%, 6·9·12개월 각 20%."""
    score = (0.4 * closes.pct_change(63, fill_method=None)
             + 0.2 * closes.pct_change(126, fill_method=None)
             + 0.2 * closes.pct_change(189, fill_method=None)
             + 0.2 * closes.pct_change(252, fill_method=None))
    return score.rank(axis=1, pct=True)


def market_ok(spy_close: pd.Series) -> pd.Series:
    return (spy_close > spy_close.rolling(50).mean()) & (spy_close > spy_close.rolling(200).mean())


def add_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    c, v = df['Close'], df['Volume']
    for n in (20, 50, 150, 200):
        df[f'SMA{n}'] = c.rolling(n).mean()
    df['Vol50Prev'] = v.rolling(50).mean().shift(1)
    df['High52'] = df['High'].rolling(250).max()
    tr = pd.concat([df['High'] - df['Low'], (df['High'] - c.shift(1)).abs(),
                    (df['Low'] - c.shift(1)).abs()], axis=1).max(axis=1)
    df['AtrRel'] = tr.rolling(14).mean() / c
    df['Trend'] = ((c > df['SMA50']) & (df['SMA50'] > df['SMA150']) & (df['SMA150'] > df['SMA200'])
                   & (df['SMA200'] > df['SMA200'].shift(20)))
    return df


def contractions(high: np.ndarray, low: np.ndarray, theta: float) -> list[tuple[int, float, float]]:
    """구간 안의 조정 목록 [(고점 위치, 고점, 저점)] — 시간순.
    theta 이상 반전해야 스윙으로 인정한다. 진행 중인 마지막 조정의 저점은 구간 끝까지의 최저가."""
    out = []
    down = False
    run_i, run_hi = 0, high[0]
    peak_i, peak, trough = 0, high[0], high[0]
    for i in range(1, len(high)):
        if down:
            trough = min(trough, low[i])
            if high[i] >= trough * (1 + theta):      # 저점 확정 → 반등 추적
                out.append((peak_i, peak, trough))
                down, run_i, run_hi = False, i, high[i]
        else:
            if high[i] > run_hi:
                run_i, run_hi = i, high[i]
            if low[i] <= run_hi * (1 - theta):       # 고점 확정 → 조정 시작
                down, peak_i, peak, trough = True, run_i, run_hi, low[i]
    if down:
        out.append((peak_i, peak, trough))
    return out


def vcp_setup(df: pd.DataFrame, i: int, p: dict = PARAMS) -> dict | None:
    """i일 종가 돌파를 가정하고, i-1일까지의 베이스가 VCP 조건을 충족하는지 판정.
    최근 base_max일의 조정들 가운데, 폭이 단계적으로 줄어드는 가장 긴 마지막 연속 구간을 베이스로 본다."""
    if i < 260:
        return None
    lo = i - p['base_max']
    high, low, vol = (df[k].to_numpy()[lo:i] for k in ('High', 'Low', 'Volume'))
    atr = df['AtrRel'].iat[i - 1]
    theta = max(p['swing_min'], 1.5 * atr) if np.isfinite(atr) else p['swing_min']
    legs = contractions(high, low, theta)
    if len(legs) < 2:
        return None
    depths = [1 - t / pk for _, pk, t in legs]
    if depths[-1] > p['last_depth_max'] or depths[-1] > depths[-2] * p['shrink']:
        return None
    k = len(depths) - 2                       # 수축이 이어지는 구간을 과거로 확장 (최대 4회)
    while k > 0 and len(depths) - k < 4 and depths[k] <= depths[k - 1] * p['shrink']:
        k -= 1
    legs, depths = legs[k:], depths[k:]
    if not (p['first_depth_min'] <= depths[0] <= p['first_depth_max']):
        return None
    base_days = len(high) - legs[0][0]
    if base_days < p['base_min']:
        return None
    pivot = float(high[legs[-1][0]:].max())
    avg50 = df['Vol50Prev'].iat[i]
    if not np.isfinite(avg50) or avg50 <= 0:
        return None
    dry = float(np.sort(vol[-10:])[:3].mean() / avg50)
    if dry > p['dryup_max']:
        return None
    return {'pivot': pivot, 'depths': [round(float(d), 4) for d in depths],
            'n_contractions': len(depths), 'dryup': round(dry, 3), 'base_days': int(base_days)}


def find_signals(df: pd.DataFrame, rs: pd.Series, mkt: pd.Series | None,
                 p: dict = PARAMS) -> list[dict]:
    """add_features() 적용된 일봉 → VCP 돌파 신호 목록. rs/mkt는 같은 날짜 인덱스의 Series."""
    c, v = df['Close'], df['Volume']
    cand = (df['Trend'] & (c >= df['High52'] * (1 - p['max_dist_high']))
            & (rs.reindex(df.index) >= p['rs_min'])
            & (v >= p['breakout_vol'] * df['Vol50Prev']) & (c > c.shift(1)))
    if mkt is not None:
        cand &= mkt.reindex(df.index).fillna(False).astype(bool)
    close = c.to_numpy()
    out = []
    for i in np.flatnonzero(cand.fillna(False).to_numpy()):
        setup = vcp_setup(df, i, p)
        if setup is None:
            continue
        pivot = setup['pivot']
        if not (close[i - 1] <= pivot < close[i] <= pivot * (1 + p['max_chase'])):
            continue
        out.append({'i': int(i), 'date': df.index[i], 'close': float(close[i]),
                    'vol_ratio': round(float(v.iat[i] / df['Vol50Prev'].iat[i]), 2),
                    'rs': round(float(rs.reindex(df.index).iat[i]), 3), **setup})
    return out


def simulate_exit(df: pd.DataFrame, i: int, p: dict = PARAMS, pivot: float | None = None) -> dict | None:
    """i일 종가 신호 → i+1일 시가 진입 후 VCP 청산 규칙 적용.
    반환 None: 데이터 부족으로 청산까지 관찰하지 못함. {'skipped': ...}: 추격 금지로 미진입."""
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ('Open', 'High', 'Low', 'Close'))
    ma = df['Close'].rolling(p['trail_ma']).mean().to_numpy()
    n, e = len(df), i + 1
    if e >= n:
        return None
    if pivot is not None and o[e] > pivot * (1 + p['max_chase']):
        return {'skipped': 'gap_above_chase_limit'}
    slip, fee = p['slippage_bps'] / 1e4, p['fee_bps'] / 1e4
    entry = o[e] * (1 + slip)
    stop, trigger = entry * (1 - p['stop_pct']), entry * (1 + p['partial_at'])
    cost = entry * (1 + fee)
    legs, remaining, partial, trail_pending, reason = [], 1.0, False, False, None
    j = e
    while j < n:
        if o[j] <= stop:
            legs.append((remaining, o[j] * (1 - slip))); reason = 'stop'; break
        if trail_pending:
            legs.append((remaining, o[j] * (1 - slip))); reason = 'trail'; break
        if j - e >= p['max_hold']:
            legs.append((remaining, o[j] * (1 - slip))); reason = 'time'; break
        if l[j] <= stop:
            legs.append((remaining, stop * (1 - slip))); reason = 'stop'; break
        if not partial and h[j] >= trigger:
            legs.append((p['partial_frac'], trigger))
            remaining -= p['partial_frac']
            partial, stop = True, entry
        if (partial or p['trail_from_start']) and np.isfinite(ma[j]) and c[j] < ma[j]:
            trail_pending = True
        j += 1
    if reason is None:
        return None
    ret = sum(w * (px * (1 - fee) / cost - 1) for w, px in legs)
    if reason == 'stop' and partial:
        reason = 'breakeven'
    return {'entry_date': str(df.index[e].date()), 'exit_date': str(df.index[j].date()),
            'entry_price': float(entry), 'rule_return': round(float(ret), 6),
            'rule_outcome': reason, 'partial': partial, 'hold_days': int(j - e + 1),
            'max_gain': round(float(h[e:j + 1].max() / entry - 1), 4),
            'execution_version': VERSION}

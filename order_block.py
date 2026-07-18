"""
오더블록(Order Block) 탐지 — v2

v1 대비 변경점:
  - 활성 약세OB 내부에 가격이 있는 경우를 'inside'로 구분 (거리 0과 혼동 방지)
  - 매매 파라미터에서 OB가 현재가 위에 있으면 정직하게 '돌파 대기'로 표기
    (v1은 '눌림목 대기'로 잘못 표기)
"""

import pandas as pd

from config import OB_MIN_MOVE, OB_LOOKBACK, OB_NEAR_TOLERANCE


def _find_obs(df: pd.DataFrame, bullish_type: bool) -> list:
    obs = []
    n = len(df)
    lookahead = 8

    for i in range(1, n - lookahead):
        candle = df.iloc[i]
        is_correct_type = (not candle['IsBull']) if bullish_type else bool(candle['IsBull'])
        if not is_correct_type or candle['BodyRatio'] < 0.005:
            continue

        for j in range(i + 1, min(i + lookahead + 1, n)):
            future = df.iloc[j]
            if bullish_type:
                denom = candle['Low']
                move = (future['High'] - denom) / denom if denom else 0
            else:
                denom = candle['High']
                move = (denom - future['Low']) / denom if denom else 0
            if move < OB_MIN_MOVE:
                continue

            has_prior = False
            for k in range(i + 1, j):
                mid = df.iloc[k]
                wrong_dir = bool(mid['IsBull']) if bullish_type else not bool(mid['IsBull'])
                if wrong_dir and mid['BodyRatio'] > 0.01:
                    has_prior = True
                    break

            if not has_prior:
                obs.append({
                    'Type': 'Bullish' if bullish_type else 'Bearish',
                    'High': float(candle['High']),
                    'Low':  float(candle['Low']),
                    'Mid':  float((candle['High'] + candle['Low']) / 2),
                    'Idx':  i,
                })
            break
    return obs


def _count_visits(df_slice: pd.DataFrame, ob_low: float, ob_high: float) -> int:
    visits, inside = 0, False
    lows, highs = df_slice['Low'].values, df_slice['High'].values
    for lo, hi in zip(lows, highs):
        in_zone = lo <= ob_high and hi >= ob_low
        if in_zone and not inside:
            visits += 1
        inside = in_zone
    return visits


def get_order_blocks(df: pd.DataFrame) -> dict:
    recent = df.iloc[-OB_LOOKBACK:] if len(df) > OB_LOOKBACK else df
    recent = recent.reset_index(drop=True)
    price  = float(df['Close'].iloc[-1])

    bull_obs = _find_obs(recent, bullish_type=True)
    bear_obs = _find_obs(recent, bullish_type=False)

    active_bull = []
    for ob in bull_obs:
        if ob['Low'] > price:          # 가격이 OB 아래 → 지지 역할 불가
            continue
        after = recent.iloc[ob['Idx'] + 1:]
        ob['visit_count'] = _count_visits(after, ob['Low'], ob['High'])
        ob['is_fresh']    = ob['visit_count'] == 0
        active_bull.append(ob)

    active_bear, flipped_bear = [], []
    for ob in bear_obs:
        if price > ob['High']:
            ob['is_flipped'] = True
            flipped_bear.append(ob)
        else:
            ob['is_flipped'] = False
            ob['price_inside'] = ob['Low'] <= price <= ob['High']
            active_bear.append(ob)

    def _dist_to_bull(ob):
        if price < ob['Low']:
            return (ob['Low'] - price) / price
        if price > ob['High']:
            return (price - ob['High']) / price
        return 0.0

    def _dist_to_flipped(ob):
        return (price - ob['High']) / price if price > ob['High'] else 0.0

    nearest_bull = min(active_bull, key=_dist_to_bull) if active_bull else None
    nearest_bull_dist = _dist_to_bull(nearest_bull) if nearest_bull else None

    nearest_flipped_dist = (min(_dist_to_flipped(ob) for ob in flipped_bear)
                            if flipped_bear else None)

    return {
        'bull':         active_bull,
        'bear':         active_bear,
        'flipped_bear': flipped_bear,
        'nearest_bull_dist':    nearest_bull_dist,
        'nearest_flipped_dist': nearest_flipped_dist,
        'near_bull': nearest_bull_dist is not None and nearest_bull_dist <= OB_NEAR_TOLERANCE,
        'bull_fresh': nearest_bull is not None and nearest_bull['is_fresh'],
    }


def calc_trade_params(ob: dict, price: float, high52w: float | None,
                      sm_score: int) -> dict:
    """OB 기반 진입가·목표가·손절가 (USD)."""
    result = {'entry': price, 'target': None, 'stop': None,
              'rr_ratio': None, 'expected_return': None,
              'entry_desc': '', 'target_desc': '', 'stop_desc': ''}

    bull_dist = ob.get('nearest_bull_dist')
    bull_obs  = sorted(ob.get('bull', []), key=lambda x: abs(price - x['Mid']))
    flip_dist = ob.get('nearest_flipped_dist')

    # 진입가
    if bull_dist == 0.0:
        result['entry_desc'] = f'강세OB 진입 중 → 현재가(${price:.2f}) 진입'
    elif flip_dist is not None and flip_dist < 0.03:
        result['entry_desc'] = 'OB플립 지지 → 현재가 진입'
    elif bull_obs and bull_dist is not None and bull_dist < 0.05:
        ob_high = bull_obs[0]['High']
        if ob_high < price:
            result['entry'] = ob_high
            result['entry_desc'] = f'강세OB 상단 눌림목 대기 ${ob_high:.2f}'
        else:
            # v1 버그 수정: OB가 현재가 위 → 눌림목이 아니라 돌파 확인 대상
            result['entry_desc'] = f'OB(${ob_high:.2f}) 돌파 확인 후 진입 권장'
    else:
        result['entry_desc'] = '현재가 진입 (근접 OB 없음)'

    entry = result['entry']

    # 목표가: 위쪽 약세OB 저항 직전 → 52주 신고가 → 기본 +10~15%
    #  - 진입가 대비 +3% 미만의 저항은 목표로 무의미 → 제외
    #  - 52주 신고가가 +35% 이상 위면 (폭락 후 회복 중) 10일 내 목표로
    #    비현실적 → 기본 목표 사용
    bear_above = sorted([b for b in ob.get('bear', [])
                         if b['Low'] > entry * 1.03],
                        key=lambda x: x['Low'])
    if bear_above:
        result['target']      = round(bear_above[0]['Low'] * 0.99, 2)
        result['target_desc'] = f'약세OB 저항(${bear_above[0]["Low"]:.2f}) 직전'
    elif high52w and entry * 1.05 < high52w <= entry * 1.35:
        result['target']      = round(high52w * 0.99, 2)
        result['target_desc'] = f'52주 신고가(${high52w:.2f}) 직전'
    else:
        pct = 0.15 if sm_score >= 80 else 0.10
        result['target']      = round(entry * (1 + pct), 2)
        result['target_desc'] = f'기본 목표 +{pct*100:.0f}%'

    # 손절가
    below = [b for b in bull_obs if b['Low'] < entry]
    if below:
        result['stop']      = round(below[0]['Low'] * 0.99, 2)
        result['stop_desc'] = f'강세OB 하단(${below[0]["Low"]:.2f}) 하회 시'
    else:
        result['stop']      = round(entry * 0.95, 2)
        result['stop_desc'] = '기본 -5% 손절'

    if result['target'] and result['stop'] and entry > result['stop']:
        profit = result['target'] - entry
        risk   = entry - result['stop']
        if risk > 0:
            result['rr_ratio']        = round(profit / risk, 2)
            result['expected_return'] = round(profit / entry * 100, 1)
    return result

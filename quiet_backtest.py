"""조용한 매집 조건 백테스트.

현재 조건(A: 60일 거래량 2x↑ + 가격 ±10%)과, 증거를 겹쳐 쌓은 강화 조건(B~E),
돌파 확인 진입(F)의 성과를 같은 매매 규칙으로 비교한다.

매매 규칙(코칭과 동일): 다음 거래일 시가 진입(비용·갭·D+3 정리 반영) → +12% 익절(고가) / 손절 3×ATR(7~15%, 저가)
/ 20거래일 종가 정리. 같은 날 익절·손절이 겹치면 손절 우선(보수적).
독립성: 종목별로 한 번 신호가 나면 20거래일 쿨다운(중복 표본 제거).
look-ahead 없음: 신호는 해당 일 종가까지의 데이터만 사용.

사용: python3 quiet_backtest.py [--refresh]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import HISTORY_PERIOD, MIN_PRICE, COACH_STOP_ATR_MULT, COACH_STOP_MIN, COACH_STOP_MAX
from data_fetcher import get_ohlcv, FetchLog
from indicators import add_all
from universe import get_ticker_list
from execution import simulate as execute, policy

CACHE = Path('/tmp/quiet_bt_cache.pkl')
BT_PERIOD = '10y'
HORIZON = 20
TARGET = 0.12
TOUCH = 0.10
COOLDOWN = 20
MIN_DOLLAR_VOL = 5_000_000


def load_data(refresh: bool = False) -> tuple[dict, pd.Series]:
    if CACHE.exists() and not refresh:
        return pickle.loads(CACHE.read_bytes())
    log = FetchLog()
    out = {}
    tickers = get_ticker_list('TRAIN')
    for i, t in enumerate(tickers, 1):
        raw = get_ohlcv(t, period=BT_PERIOD, log=log)
        if len(raw) >= 260:
            out[t] = raw
        if i % 25 == 0:
            print(f'  fetch {i}/{len(tickers)}', end='\r')
    spy = get_ohlcv('SPY', period=BT_PERIOD, log=log)['Close']
    CACHE.write_bytes(pickle.dumps((out, spy)))
    return out, spy


def features(raw: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    df = add_all(raw.copy())
    c, v = df['Close'], df['Volume']
    # B: 하루짜리 이벤트 제외 — 60일 최대 거래량 / 60일 평균
    df['MaxVolRel'] = v.rolling(60).max() / v.rolling(60).mean().replace(0, np.nan)
    # C: 매집 방향 — 상승일 거래량 합 / 하락일 거래량 합(60일), OBV 60일 신고점 근접
    up = (v * (c.diff() > 0)).rolling(60).sum()
    dn = (v * (c.diff() < 0)).rolling(60).sum()
    df['UpDnVol'] = up / dn.replace(0, np.nan)
    df['OBVNearHigh'] = (df['OBV'] >= df['OBV'].rolling(60).max() * 0.98) | (df['OBV'] >= df['OBV'].rolling(60).max() - 0.02 * df['Volume'].rolling(60).mean() * 60)
    # D: 상대강도(20일, vs SPY) + 추세 위
    spy20 = spy.reindex(df.index).ffill().pct_change(20)
    df['RS20'] = c.pct_change(20) - spy20
    # E: 변동성 수축 — 현재 ATR%가 60일 전보다 작음
    df['ATRShrink'] = df['ATRRel'] < df['ATRRel'].shift(60)
    df['BoxPos'] = (c - c.rolling(60).min()) / (c.rolling(60).max() - c.rolling(60).min()).replace(0, np.nan)
    df['DollarVol'] = (c * df['VolMA'])
    df['Hi60prev'] = df['High'].rolling(60).max().shift(1)
    return df


def conditions(df: pd.DataFrame) -> dict[str, pd.Series]:
    base = (df['Close'] >= MIN_PRICE) & (df['DollarVol'] >= MIN_DOLLAR_VOL)
    A = base & (df['VolSurge3M'] >= 2.0) & (df['Price3M'].abs() < 0.10)
    B = A & (df['MaxVolRel'] <= 5.0)
    C = B & (df['UpDnVol'] >= 1.3) & df['OBVNearHigh']
    D = C & (df['RS20'] >= 0) & (df['Close'] > df['MA200'])
    E = D & df['ATRShrink'] & (df['BoxPos'] >= 0.5)
    # F: 매집 상태(C) 이후 20일 안에 박스 상단 돌파 + 거래량 1.5x 확인 시 진입
    state = C.rolling(HORIZON).max().shift(1).fillna(0).astype(bool)
    F = base & state & (df['Close'] > df['Hi60prev']) & (df['VolRatio'] >= 1.5)
    # G: 돌파 확인 + 추세/상대강도 필터까지
    G = F & (df['RS20'] >= 0) & (df['Close'] > df['MA200'])
    abl = {'A+이벤트제외': A & (df['MaxVolRel'] <= 5.0),
           'A+상승/하락거래량': A & (df['UpDnVol'] >= 1.3),
           'A+OBV신고점': A & df['OBVNearHigh'],
           'A+RS≥0': A & (df['RS20'] >= 0),
           'A+MA200위': A & (df['Close'] > df['MA200']),
           'A+ATR수축': A & df['ATRShrink'],
           'A+박스상단': A & (df['BoxPos'] >= 0.5),
           'A+돌파확인': base & A.rolling(HORIZON).max().shift(1).fillna(0).astype(bool)
                          & (df['Close'] > df['Hi60prev']) & (df['VolRatio'] >= 1.5)}
    return {**abl, 'ALL(기준선)': base, 'A 현재': A, 'B +이벤트제외': B, 'C +매집방향': C,
            'D +RS·추세': D, 'E +수축·박스상단': E, 'F 돌파확인': F, 'G 돌파+RS·추세': G}


def simulate(df: pd.DataFrame, mask: pd.Series) -> list[dict]:
    idx = np.where(mask.values)[0]
    c, h, l = df['Close'].values, df['High'].values, df['Low'].values
    atr = df['ATRRel'].values
    out, last = [], -10**9
    for i in idx:
        if i - last < COOLDOWN or i + HORIZON >= len(df) or np.isnan(atr[i]):
            continue
        last = i
        entry = c[i]
        ev = execute(df.iloc[i+1:i+HORIZON+1], HORIZON, policy(atr[i]))
        if ev is None:
            continue
        res, ret = ev['rule_outcome'], ev['rule_return']
        touch = h[i + 1:i + HORIZON + 1].max() / entry - 1 >= TOUCH
        out.append({'date': df.index[i], 'res': res, 'ret': ret, 'touch': touch,
                    'mdd': l[i + 1:i + HORIZON + 1].min() / entry - 1})
    return out


def summarize(name: str, rows: list[dict]) -> str:
    if not rows:
        return f'{name:<18} n=0'
    r = pd.DataFrame(rows).sort_values('date')
    half = len(r) // 2
    h1, h2 = r.iloc[:half], r.iloc[half:]
    f = lambda d: f'{(d.ret > 0).mean() * 100:.0f}%/{d.ret.mean() * 100:+.1f}%' if len(d) else '-'
    return (f"{name:<18} n={len(r):4d} 터치{r.touch.mean() * 100:3.0f}% 승률{(r.ret > 0).mean() * 100:3.0f}% "
            f"평균{r.ret.mean() * 100:+5.1f}% 목표{(r.res == 'target').mean() * 100:3.0f}% "
            f"손절{(r.res == 'stop').mean() * 100:3.0f}% 기한{(r.res == 'time').mean() * 100:3.0f}% "
            f"| 전반 {f(h1)} 후반 {f(h2)}")


def main():
    data, spy = load_data('--refresh' in sys.argv)
    print(f'종목 {len(data)}개, 기간 {BT_PERIOD}')
    agg: dict[str, list[dict]] = {}
    for t, raw in data.items():
        df = features(raw, spy)
        for name, m in conditions(df).items():
            agg.setdefault(name, []).extend(simulate(df, m.fillna(False)))
    print('규칙: 진입 다음 시가 / 익절 +12% / 손절 3×ATR(7~15%) / 20일 정리, 쿨다운 20일')
    for name, rows in agg.items():
        print(summarize(name, rows))


if __name__ == '__main__':
    main()

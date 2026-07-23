"""
기술적 지표 (v2) — 전 구간 벡터화 계산

v1의 치명적 문제 해결:
  1. VCP / OBV_Diverge가 마지막 행에만 계산되던 학습-실전 불일치(train/serve skew)
     → 모든 행에 대해 벡터화 계산. 학습 데이터와 실전 예측이 동일한 값을 본다.
  2. 모델 피처 컬럼명 불일치 (RangeRel, OBVSlope5, StochK14, PriceVsMA20,
     MA20AboveMA50, MA50AboveMA200, Dist52WPct가 계산되지 않아 항상 NaN)
     → 여기서 전부 실제로 계산하고, MODEL_FEATURES로 명시.
     → validate_features()가 누락/전체-NaN 컬럼을 학습·예측 시점에 즉시 에러로 알림.
  3. MA 체계를 태그와 일치하는 US 관행(20/50/200)으로 통일.

모든 지표는 해당 시점까지의 데이터만 사용한다 (look-ahead 없음).
"""

import numpy as np
import pandas as pd

from config import MA_PERIODS, VOLUME_AVG_PERIOD

# 모델이 사용하는 피처 목록 — add_all()이 전부 생성함을 validate_features()가 보증
MODEL_FEATURES = [
    'VolRatio', 'VolSlope5', 'VolDecDays', 'VolRecovery',
    'RangeRel', 'ATRRel', 'ATRCompress', 'BBWidth', 'BBCompress', 'NearBBLower',
    'NR7', 'NRStreak', 'VCP',
    'OBVSlope5', 'OBV_Diverge',
    'RSI14', 'RSI_Recovery', 'StochK14',
    'PriceVsMA20', 'MA20AboveMA50', 'MA50AboveMA200', 'GoldenCross',
    'Dist52WPct', 'IsBull', 'BodyRatio',
    'Gain5D', 'Gain10D',
]

TREND_FEATURES = ['VolRatio', 'RangeRel', 'ATRCompress', 'BBCompress', 'RSI14']


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _basic(df)
    df = _moving_averages(df)
    df = _volume(df)
    df = _volatility(df)
    df = _nr7_vcp(df)
    df = _obv(df)
    df = _oscillators(df)
    df = _high_52w(df)
    return df


def validate_features(df: pd.DataFrame, features: list[str] | None = None,
                      optional: set[str] | None = None) -> None:
    """모델 피처가 실제로 존재하고 전체 NaN이 아닌지 검증. 실패 시 즉시 에러.
    (v1은 피처 7개가 조용히 NaN이었다 — 다시는 조용히 죽지 않게 한다.)

    optional: 전체 NaN을 허용하는 컬럼 집합 — 외부 소스라 일시 수집 실패가
    있을 수 있는 매크로 피처용. 존재 자체는 여전히 필수이고, 전체 NaN이면
    에러 대신 경고를 출력한다 (조용히 삼키지 않되 스캔은 막지 않음)."""
    features = features or MODEL_FEATURES
    optional = optional or set()
    missing  = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f'지표 컬럼 누락: {missing}')
    all_nan = [c for c in features
               if df[c].isna().all()]
    hard = [c for c in all_nan if c not in optional]
    if hard:
        raise ValueError(f'지표 컬럼이 전부 NaN: {hard}')
    soft = [c for c in all_nan if c in optional]
    if soft:
        print(f'  ⚠ 선택 피처 전부 NaN (수집 실패 — 모델이 결측 처리): {soft}')


# ── 기본 속성 ─────────────────────────────────────────────────────────────────

def _basic(df):
    close = df['Close']
    df['Change']    = close.pct_change()
    df['IsBull']    = (close >= df['Open']).astype(float)
    df['BodyRatio'] = (close - df['Open']).abs() / close.replace(0, np.nan)
    df['Gain5D']    = close.pct_change(5)
    df['Gain10D']   = close.pct_change(10)
    return df


# ── 이동평균 (US 관행 20/50/200) ──────────────────────────────────────────────

def _moving_averages(df):
    close = df['Close']
    for p in MA_PERIODS:
        df[f'MA{p}'] = close.rolling(p).mean()

    ma20, ma50, ma200 = df['MA20'], df['MA50'], df['MA200']
    df['PriceVsMA20']    = (close - ma20) / ma20.replace(0, np.nan)
    df['MA20AboveMA50']  = (ma20 > ma50).astype(float)
    df['MA50AboveMA200'] = (ma50 > ma200).astype(float)
    df['MaBull'] = (ma20 > ma50) & (ma50 > ma200)
    df['MaBear'] = (ma20 < ma50) & (ma50 < ma200)
    df['GoldenCross'] = ((ma20.shift(1) <= ma50.shift(1)) & (ma20 > ma50)).astype(float)
    return df


# ── 거래량 ────────────────────────────────────────────────────────────────────

def _volume(df):
    v = df['Volume']
    df['VolMA']    = v.rolling(VOLUME_AVG_PERIOD).mean()
    df['VolRatio'] = (v / df['VolMA'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # 5일 거래량비 기울기 — 고정 가중치 최소자승 (벡터화)
    r = df['VolRatio']
    df['VolSlope5'] = (2 * r + r.shift(1) - r.shift(3) - 2 * r.shift(4)) / 10.0

    # 연속 거래량 감소일 수
    down = (v < v.shift(1)).astype(int)
    grp  = (down != down.shift()).cumsum()
    streak = down.groupby(grp).cumcount() + 1
    df['VolDecDays'] = (streak * down).astype(float)

    # 압축 후 회복: 직전일까지 3일+ 연속 감소 후 오늘 거래량비 반등
    had_compression = df['VolDecDays'].shift(1) >= 3
    now_expanding   = r >= r.shift(1)
    df['VolRecovery'] = (had_compression & now_expanding).astype(float)
    return df


# ── 변동성 (Range / ATR / 볼린저) ─────────────────────────────────────────────

def _volatility(df):
    h, l, c = df['High'], df['Low'], df['Close']
    df['RangeRel'] = (h - l) / c.replace(0, np.nan)

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df['ATR14']       = tr.rolling(14).mean()
    df['ATRRel']      = df['ATR14'] / c.replace(0, np.nan)
    df['ATRCompress'] = df['ATR14'] / df['ATR14'].rolling(20).mean().replace(0, np.nan)

    ma20   = df['MA20'] if 'MA20' in df.columns else c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df['BBWidth']    = (4 * bb_std) / ma20.replace(0, np.nan)
    df['BBCompress'] = df['BBWidth'] / df['BBWidth'].rolling(20).mean().replace(0, np.nan)
    bb_lower = ma20 - 2 * bb_std
    df['NearBBLower'] = ((c - bb_lower) / bb_lower.replace(0, np.nan) < 0.03).astype(float)
    return df


# ── NR7 / VCP (전 구간 벡터화) ────────────────────────────────────────────────

def _nr7_vcp(df):
    rng = df['High'] - df['Low']
    roll_min = rng.rolling(7).min()
    df['NR7'] = ((rng <= roll_min) & roll_min.notna()).astype(float)

    nr = df['NR7'].astype(int)
    grp = (nr != nr.shift()).cumsum()
    streak = nr.groupby(grp).cumcount() + 1
    df['NRStreak'] = (streak * nr).astype(float)

    # VCP: 최근 20일을 5일 구간 4개로 나눠 각 구간 변동폭이 연속 감소
    # (v1은 마지막 행에만 계산 → 학습 데이터에서 항상 False였던 버그 수정)
    r5 = df['High'].rolling(5).max() - df['Low'].rolling(5).min()
    s0, s1, s2, s3 = r5, r5.shift(5), r5.shift(10), r5.shift(15)
    df['VCP'] = ((s0 < s1) & (s1 < s2) & (s2 < s3)).astype(float)
    return df


# ── OBV (전 구간 벡터화) ──────────────────────────────────────────────────────

def _obv(df):
    direction = np.sign(df['Close'].diff()).fillna(0)
    df['OBV'] = (direction * df['Volume']).cumsum()

    # OBV 5일 기울기 — 평균 거래량으로 정규화한 무차원 값
    vol_ma = df['VolMA'] if 'VolMA' in df.columns else df['Volume'].rolling(VOLUME_AVG_PERIOD).mean()
    df['OBVSlope5'] = df['OBV'].diff(5) / (5 * vol_ma.replace(0, np.nan))

    # 다이버전스: 20일 가격 변화 < +3% 인데 OBV는 증가 = 조용한 매집
    # (v1은 마지막 행에만 계산 → 전 구간 벡터화)
    price_chg20 = df['Close'].pct_change(20)
    obv_chg20   = df['OBV'].diff(20)
    df['OBV_Diverge'] = ((price_chg20 < 0.03) & (obv_chg20 > 0)).astype(float)
    return df


# ── RSI / Stochastic ─────────────────────────────────────────────────────────

def _oscillators(df, period: int = 14):
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI14'] = 100 - (100 / (1 + rs))
    df['RSI_Recovery'] = df['RSI14'].between(30, 55).astype(float)

    low14  = df['Low'].rolling(period).min()
    high14 = df['High'].rolling(period).max()
    rng = (high14 - low14).replace(0, np.nan)
    df['StochK14'] = (df['Close'] - low14) / rng * 100
    return df


# ── 52주 신고가 ───────────────────────────────────────────────────────────────

def _high_52w(df):
    lookback = min(250, len(df))
    df['High52W'] = df['High'].rolling(lookback, min_periods=lookback).max()
    dist = (df['High52W'] - df['Close']) / df['High52W'].replace(0, np.nan)
    df['Dist52WPct'] = dist.clip(lower=0)
    return df

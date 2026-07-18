"""
데이터 수집 (yfinance) — v2

v1 대비 변경점:
  - 실패를 조용히 삼키지 않는다: fetch 실패는 FetchLog에 기록되어
    스캔 리포트에 "수집 실패 종목" 목록으로 출력된다.
  - info 실패 시 기관보유/공매도를 0으로 조작하지 않고 data_ok=False로 표시
    → 스코어러가 해당 항목을 '데이터 없음'으로 처리 (0점 조작과 구분).
  - 공매도 비율(shortPercentOfFloat)은 격주 갱신 데이터임을 신호 문자열에 명시.
"""

import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf


class FetchLog:
    """스캔 한 번 동안의 데이터 수집 실패 기록."""
    def __init__(self):
        self.ohlcv_fail: list[tuple[str, str]] = []
        self.info_fail:  list[str] = []

    def summary_lines(self) -> list[str]:
        lines = []
        if self.ohlcv_fail:
            names = ', '.join(t for t, _ in self.ohlcv_fail[:20])
            lines.append(f'  ⚠ OHLCV 수집 실패 {len(self.ohlcv_fail)}종목: {names}')
        if self.info_fail:
            names = ', '.join(self.info_fail[:20])
            lines.append(f'  ⚠ 재무정보 수집 실패 {len(self.info_fail)}종목 (기관/공매도 점수 제외됨): {names}')
        if not lines:
            lines.append('  데이터 수집: 전 종목 정상')
        return lines


def get_ohlcv(ticker: str, period: str = '2y',
              end_date: str | None = None,
              log: FetchLog | None = None) -> pd.DataFrame:
    """OHLCV 수집. end_date='YYYY-MM-DD'면 해당 종가까지."""
    try:
        if end_date:
            end_dt   = datetime.strptime(end_date, '%Y-%m-%d')
            years    = int(period.rstrip('y')) if period.endswith('y') else 2
            start_dt = end_dt - timedelta(days=365 * years + 30)
            df = yf.download(ticker,
                             start=start_dt.strftime('%Y-%m-%d'),
                             end=(end_dt + timedelta(days=1)).strftime('%Y-%m-%d'),
                             progress=False, auto_adjust=True, actions=False)
        else:
            df = yf.download(ticker, period=period, progress=False,
                             auto_adjust=True, actions=False)

        if df is None or df.empty:
            raise ValueError('empty dataframe')

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.index.name = 'Date'
        df = df.dropna(subset=['Close', 'Volume'])
        df = df[df['Volume'] > 0]
        return df
    except Exception as e:
        if log is not None:
            log.ohlcv_fail.append((ticker, str(e)[:60]))
        return pd.DataFrame()


def get_info(ticker: str, log: FetchLog | None = None) -> dict:
    """재무·공매도·기관보유. 실패 시 data_ok=False (0으로 조작하지 않음)."""
    result = {
        'data_ok': False,
        'name': ticker, 'sector': '', 'industry': '',
        'market_cap': 0, 'pe_ratio': None,
        'short_pct': None, 'inst_pct': None,
    }
    try:
        info = yf.Ticker(ticker).info
        if not info or 'longName' not in info:
            raise ValueError('empty info')
        short = info.get('shortPercentOfFloat')
        inst  = info.get('heldPercentInstitutions')
        result.update({
            'data_ok':    True,
            'name':       info.get('longName', ticker),
            'sector':     info.get('sector', ''),
            'industry':   info.get('industry', ''),
            'market_cap': info.get('marketCap', 0) or 0,
            'pe_ratio':   info.get('trailingPE'),
            'short_pct':  short * 100 if short is not None else None,
            'inst_pct':   inst * 100 if inst is not None else None,
        })
    except Exception:
        if log is not None:
            log.info_fail.append(ticker)
    return result


def short_signal(info: dict) -> str:
    """공매도 신호 문자열. 데이터 특성(격주 갱신, 최대 2주 지연)을 명시."""
    sp = info.get('short_pct')
    if sp is None:
        return ''
    if sp >= 20:
        return f'공매도 {sp:.0f}% — 쇼트스퀴즈 후보 (※격주 갱신 데이터, 최대 2주 지연)'
    if sp >= 10:
        return f'공매도 {sp:.0f}% 높음 (※격주 갱신 데이터)'
    return f'공매도 {sp:.0f}%'


def fmt_market_cap(cap: float) -> str:
    if not cap:
        return '-'
    if cap >= 1e12: return f'${cap/1e12:.1f}T'
    if cap >= 1e9:  return f'${cap/1e9:.0f}B'
    if cap >= 1e6:  return f'${cap/1e6:.0f}M'
    return f'${cap:,.0f}'

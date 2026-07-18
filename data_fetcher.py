"""
데이터 수집 — v2

이중 경로:
  1차: yfinance (일반 PC 환경)
  2차: Yahoo Finance API 직접 호출 (requests 기반 폴백)
       — yfinance가 내부적으로 쓰는 curl_cffi의 브라우저 TLS 위장이
         보안 프록시/방화벽 환경에서 차단되는 경우가 있다 (Claude Code
         샌드박스 포함). 그 경우 표준 requests로 동일 데이터를 수집한다.
       - OHLCV: /v8/finance/chart (adjclose로 수정주가 반영)
       - 재무/공매도/기관: /v10/finance/quoteSummary (cookie+crumb 인증)

v1 대비 변경점:
  - 실패를 조용히 삼키지 않는다: 실패는 FetchLog에 기록되어 리포트에 출력.
  - info 실패 시 0으로 조작하지 않고 data_ok=False → 스코어러가 항목 제외.
  - 공매도 비율은 격주 갱신 데이터임을 신호 문자열에 명시.
"""

import time
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import requests as _rq

_UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                     'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'}
_session: _rq.Session | None = None
_crumb: str | None = None
_yf_broken = False   # yfinance 1회 실패 + 폴백 성공 시 True → 이후 폴백 직행 (시간 절약)


class FetchLog:
    """스캔 한 번 동안의 데이터 수집 실패 기록."""
    def __init__(self):
        self.ohlcv_fail: list[tuple[str, str]] = []
        self.info_fail:  list[str] = []
        self.fallback_used = False

    def summary_lines(self) -> list[str]:
        lines = []
        if self.fallback_used:
            lines.append('  ℹ yfinance 차단 감지 → Yahoo API 직접 호출(폴백)로 수집함')
        if self.ohlcv_fail:
            names = ', '.join(t for t, _ in self.ohlcv_fail[:20])
            lines.append(f'  ⚠ OHLCV 수집 실패 {len(self.ohlcv_fail)}종목: {names}')
        if self.info_fail:
            names = ', '.join(self.info_fail[:20])
            lines.append(f'  ⚠ 재무정보 수집 실패 {len(self.info_fail)}종목 '
                         f'(기관/공매도 점수 제외됨): {names}')
        if not lines:
            lines.append('  데이터 수집: 전 종목 정상')
        return lines


# ── Yahoo API 직접 호출 (폴백) ────────────────────────────────────────────────

def _get_session() -> _rq.Session:
    global _session
    if _session is None:
        _session = _rq.Session()
        _session.headers.update(_UA)
    return _session


def _get_crumb(refresh: bool = False) -> str | None:
    global _crumb
    if _crumb is not None and not refresh:
        return _crumb
    try:
        s = _get_session()
        s.get('https://fc.yahoo.com', timeout=15)
        r = s.get('https://query1.finance.yahoo.com/v1/test/getcrumb', timeout=15)
        _crumb = r.text.strip() if r.status_code == 200 and r.text else None
    except Exception:
        _crumb = None
    return _crumb


def _api_get(url: str, params: dict, retries: int = 2) -> _rq.Response | None:
    s = _get_session()
    for attempt in range(retries + 1):
        try:
            r = s.get(url, params=params, timeout=20)
            if r.status_code == 429:            # 레이트리밋 → 백오프 후 재시도
                time.sleep(2 * (attempt + 1))
                continue
            return r
        except Exception:
            if attempt == retries:
                return None
            time.sleep(1)
    return None


def _chart_ohlcv(ticker: str, period: str, end_date: str | None) -> pd.DataFrame:
    """/v8/finance/chart 기반 OHLCV. adjclose 비율로 수정주가 반영."""
    params = {'interval': '1d', 'events': 'div,split'}
    if end_date:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        years  = int(period.rstrip('y')) if period.endswith('y') else 2
        start_dt = end_dt - timedelta(days=365 * years + 30)
        params['period1'] = int(start_dt.timestamp())
        params['period2'] = int(end_dt.timestamp())
    else:
        params['range'] = period

    r = _api_get(f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}', params)
    if r is None or r.status_code != 200:
        raise ValueError(f'chart api {r.status_code if r is not None else "no-response"}')
    result = r.json().get('chart', {}).get('result')
    if not result:
        raise ValueError('chart api empty result')
    d = result[0]
    ts = d.get('timestamp')
    q  = d.get('indicators', {}).get('quote', [{}])[0]
    if not ts or not q.get('close'):
        raise ValueError('chart api no bars')

    idx = (pd.to_datetime(ts, unit='s', utc=True)
             .tz_convert('America/New_York').normalize().tz_localize(None))
    df = pd.DataFrame({
        'Open': q['open'], 'High': q['high'], 'Low': q['low'],
        'Close': q['close'], 'Volume': q['volume'],
    }, index=idx)

    adj = d.get('indicators', {}).get('adjclose', [{}])[0].get('adjclose')
    if adj is not None:
        factor = pd.Series(adj, index=idx) / df['Close']
        for col in ('Open', 'High', 'Low', 'Close'):
            df[col] = df[col] * factor
    df.index.name = 'Date'
    return df


def _quote_summary(ticker: str) -> dict:
    """/v10/finance/quoteSummary 기반 재무·공매도·기관 (crumb 인증)."""
    modules = 'defaultKeyStatistics,summaryProfile,price,summaryDetail'
    for refresh in (False, True):
        crumb = _get_crumb(refresh=refresh)
        if crumb is None:
            break
        r = _api_get(f'https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}',
                     {'modules': modules, 'crumb': crumb})
        if r is not None and r.status_code == 200:
            res = r.json().get('quoteSummary', {}).get('result')
            if res:
                return res[0]
        if r is not None and r.status_code != 401:
            break                              # 401만 crumb 갱신 재시도
    raise ValueError('quoteSummary failed')


def _raw(d: dict, *keys):
    cur = d
    for k in keys:
        cur = cur.get(k, {}) if isinstance(cur, dict) else {}
    return cur.get('raw') if isinstance(cur, dict) else None


# ── 공개 API ─────────────────────────────────────────────────────────────────

def get_ohlcv(ticker: str, period: str = '2y',
              end_date: str | None = None,
              log: FetchLog | None = None) -> pd.DataFrame:
    """OHLCV 수집. end_date='YYYY-MM-DD'면 해당 종가까지.
    yfinance → 실패 시 Yahoo API 직접 호출 폴백."""
    global _yf_broken
    df = pd.DataFrame()

    # 1차: yfinance (이전에 차단 확인되면 건너뜀)
    try:
        if _yf_broken:
            raise RuntimeError('yfinance unavailable (fallback active)')
        import yfinance as yf
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
        if df is not None and not df.empty and isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception:
        df = pd.DataFrame()

    # 2차: Yahoo API 직접 호출
    if df is None or df.empty:
        try:
            df = _chart_ohlcv(ticker, period, end_date)
            _yf_broken = True
            if log is not None:
                log.fallback_used = True
            time.sleep(0.1)                    # 폴백 경로는 예의상 간격 유지
        except Exception as e:
            if log is not None:
                log.ohlcv_fail.append((ticker, str(e)[:60]))
            return pd.DataFrame()

    try:
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

    # 1차: yfinance
    try:
        if _yf_broken:
            raise RuntimeError('yfinance unavailable (fallback active)')
        import yfinance as yf
        info = yf.Ticker(ticker).info
        if info and 'longName' in info:
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
            return result
    except Exception:
        pass

    # 2차: quoteSummary 직접 호출
    try:
        qs = _quote_summary(ticker)
        prof  = qs.get('summaryProfile', {}) or {}
        price = qs.get('price', {}) or {}
        short = _raw(qs, 'defaultKeyStatistics', 'shortPercentOfFloat')
        inst  = _raw(qs, 'defaultKeyStatistics', 'heldPercentInstitutions')
        result.update({
            'data_ok':    True,
            'name':       price.get('longName') or ticker,
            'sector':     prof.get('sector', '') or '',
            'industry':   prof.get('industry', '') or '',
            'market_cap': _raw(qs, 'price', 'marketCap') or 0,
            'pe_ratio':   _raw(qs, 'summaryDetail', 'trailingPE'),
            'short_pct':  short * 100 if short is not None else None,
            'inst_pct':   inst * 100 if inst is not None else None,
        })
        time.sleep(0.1)
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

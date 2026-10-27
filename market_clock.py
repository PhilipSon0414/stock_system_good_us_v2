"""NYSE sessions, including DST, holidays and early closes."""
from functools import lru_cache
import pandas as pd
import numpy as np
import exchange_calendars as xcals

@lru_cache(maxsize=1)
def calendar():
    return xcals.get_calendar('XNYS', start='1990-01-01', end='2035-12-31')

def utc_now(now=None):
    t = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if t.tzinfo is None:
        raise ValueError('Execution time must include a timezone')
    return t.tz_convert('UTC')

def completed_session(now=None, delay_minutes=30):
    cutoff = utc_now(now) - pd.Timedelta(minutes=delay_minutes)
    cal = calendar()
    day = cutoff.tz_convert('America/New_York').strftime('%Y-%m-%d')
    session = cal.date_to_session(day, direction='previous')
    if cal.session_close(session) > cutoff:
        session = cal.previous_session(session)
    return session.strftime('%Y-%m-%d')

def resolve_session(requested=None, now=None):
    latest = completed_session(now)
    if requested is None:
        return latest
    requested = pd.Timestamp(requested).strftime('%Y-%m-%d')
    if not calendar().is_session(requested) or requested > latest:
        raise ValueError('기준일은 완료된 미국 거래일이어야 합니다: ' + requested)
    return requested

def session_offset(day, offset):
    return calendar().session_offset(pd.Timestamp(day), offset).strftime('%Y-%m-%d')

def session_dates(start, end):
    return calendar().sessions_in_range(start, end).tz_localize(None)

def completed_bars(df, session):
    """Reject stale/incomplete data; never silently substitute an older session."""
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert('America/New_York').tz_localize(None)
    df.index = idx.normalize()
    df = df.loc[df.index <= pd.Timestamp(session)].sort_index()
    if df.empty or df.index.has_duplicates or df.index[-1] != pd.Timestamp(session):
        raise ValueError('기준 거래일 일봉 누락/중복: ' + session)
    expected = session_dates(df.index[0], session)
    if not df.index.equals(expected):
        raise ValueError('거래일 시계열 누락/비거래일 포함')
    values = df[['Open','High','Low','Close','Volume']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('OHLCV 불완전')
    if ((df.High < df[['Open','Close']].max(axis=1)) |
        (df.Low > df[['Open','Close']].min(axis=1))).any():
        raise ValueError('OHLC 가격 범위 불일치')
    return df

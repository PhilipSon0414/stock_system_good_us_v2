"""
매크로 선행지표 (v2) — 달러 인덱스 / 하이일드 스프레드 / 구리

"증시를 선제적으로 반영한다"는 3계열을 미국 시장 기준으로 실검증 후 편입.

사전 검증 (2010~2026, S&P500 미래 10일 수익률 대비 스피어만 IC —
python3 macro.py check 로 재현):

  하이일드 스프레드 1년 z-score  IC +0.26  (Q1 +0.04% vs Q5 +1.97%)
    → 스프레드가 1년 범위의 고점(공포 정점)일 때 미래 수익률이 가장 높다.
      '스프레드 확대=위험' 논지는 방향 전환(chg20 IC -0.04)보다
      수준(level) 역발상 신호로 작동한다. 셋 중 유일하게 강한 신호.
  달러 인덱스 20일 속도          IC ~0.00  (미국 시장엔 무신호)
    → '달러 강세=자금 이탈'은 신흥국 논리. 같은 검증을 KOSPI에 하면
      IC -0.11이 나오지만 S&P500엔 나오지 않는다 (미국은 달러의 본국).
  구리 20일 모멘텀               IC -0.05  (미국 시장엔 무신호~미약한 역)
    → '닥터 코퍼' 역시 제조업/신흥국 지수에 민감. KOSPI IC +0.07 대비
      S&P500에는 유의미한 선행성이 없다.

결론: 3지표 서사는 대체로 신흥국 이야기이고, 미국 시장에서 실제로
선행하는 것은 하이일드 스프레드(공포 수준)다. 그래도 세 계열 모두
피처로 공급한다 — GBM이 상호작용(예: 공포 국면에서 셋업 신호 재해석)을
학습하거나 무시하도록 두고, 하드 룰은 만들지 않는다.

누수 차단: 세 계열 모두 미국 데이터라 시차 문제는 한국판보다 단순하다.
장 마감 후 스캔 시점 기준 '어제까지' 값만 쓰도록 종목 캘린더에 ffill 정렬
후 1일 shift. (HY OAS는 익일 아침 발표라 shift(1)이 정확히 발표 시점과 일치)

사용:
  python3 macro.py check    # S&P500 선행성 검증 리포트 (IC/5분위)
"""

import io
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import requests

MACRO_FEATURES = [
    'dxy_ret5', 'dxy_ret20', 'dxy_accel',
    'hy_chg5', 'hy_chg20', 'hy_z252',
    'copper_ret5', 'copper_ret20', 'copper_rebound',
]

_FRED_CSV = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2'

_frame: pd.DataFrame | None = None      # 프로세스당 1회 수집 (스캔/학습 공용)
_fetch_failed = False


# ── 수집 ─────────────────────────────────────────────────────────────────────

def _yahoo_close(ticker: str, years: int = 5) -> pd.Series:
    """야후 chart API로 종가 시리즈. data_fetcher.get_ohlcv를 쓰지 않는 이유:
    지수(DXY)는 거래량이 0이라 get_ohlcv의 Volume>0 필터에 걸려 빈 값이 된다."""
    from data_fetcher import _chart_ohlcv
    df = _chart_ohlcv(ticker, f'{years}y' if years in (1, 2, 5, 10) else 'max', None)
    return df['Close'].astype(float).dropna()


def _fred_hy() -> pd.Series:
    """ICE BofA US High Yield OAS (%). FRED CSV — API 키 불필요."""
    r = requests.get(_FRED_CSV, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), na_values='.')
    date_col, val_col = df.columns[0], df.columns[1]
    s = pd.Series(df[val_col].astype(float).values,
                  index=pd.to_datetime(df[date_col]))
    return s.dropna()


def _build_frame() -> pd.DataFrame:
    """원시 3계열 → 피처 프레임 (각 계열 자기 캘린더에서 계산 후 합집합)."""
    feat = {}
    d = _yahoo_close('DX-Y.NYB')
    feat['dxy_ret5'] = d.pct_change(5)
    feat['dxy_ret20'] = d.pct_change(20)
    # '방향이 아니라 속도': 변화율의 변화 — 달러 꺾임 감지
    feat['dxy_accel'] = d.pct_change(5) - d.pct_change(5).shift(5)

    h = _fred_hy()
    feat['hy_chg5'] = h.diff(5)                      # %p 변화
    feat['hy_chg20'] = h.diff(20)
    feat['hy_z252'] = ((h - h.rolling(252, min_periods=120).mean())
                       / h.rolling(252, min_periods=120).std())

    c = _yahoo_close('HG=F')
    feat['copper_ret5'] = c.pct_change(5)
    feat['copper_ret20'] = c.pct_change(20)
    # 닥터 코퍼 바닥 반등: 60일 저점 대비 +5% & 20일선 위
    feat['copper_rebound'] = (
        ((c / c.rolling(60).min() - 1 > 0.05)
         & (c > c.rolling(20).mean())).astype(float))
    return pd.DataFrame(feat)


def get_macro_frame(verbose: bool = False) -> pd.DataFrame | None:
    """피처 프레임 (메모이즈). 수집 실패 시 None — 호출자가 명시 표기
    (조용한 0/NaN 삼킴 금지, 이 리포의 원칙)."""
    global _frame, _fetch_failed
    if _frame is not None:
        return _frame
    if _fetch_failed:
        return None
    try:
        _frame = _build_frame()
        if verbose:
            print(f'  매크로 지표 수집: {len(_frame.columns)}피처 '
                  f'(~{_frame.index[-1].date()})')
        return _frame
    except Exception as e:
        _fetch_failed = True
        print(f'  ⚠ 매크로 지표 수집 실패 — 피처 NaN으로 진행: {str(e)[:80]}')
        return None


def add_macro(df: pd.DataFrame) -> pd.DataFrame:
    """종목 DataFrame(날짜 인덱스)에 매크로 피처 병합.

    종목 캘린더로 reindex+ffill 후 1일 shift — 스캔 시점(장 마감 후)에
    확정된 '어제까지' 값만 사용 (look-ahead 차단).
    수집 실패 시 NaN 컬럼 (HistGBM이 네이티브 처리, validate에서 optional).
    """
    mac = get_macro_frame()
    if mac is None:
        for col in MACRO_FEATURES:
            df[col] = np.nan
        return df
    aligned = (mac.reindex(mac.index.union(df.index)).ffill()
               .reindex(df.index).shift(1))
    for col in MACRO_FEATURES:
        df[col] = aligned[col]
    return df


# ── 리포트용 국면 요약 ────────────────────────────────────────────────────────

def regime_line() -> str | None:
    """스캔 리포트 헤더용 한 줄. 검증 결과에 따라 HY 스프레드가 주 신호,
    달러/구리는 참고 표기. 수집 실패 시 None."""
    mac = get_macro_frame()
    if mac is None or mac.empty:
        return None
    r = mac.ffill().iloc[-1]

    def _f(col, pct=True):
        v = r.get(col)
        if v is None or v != v:
            return 'N/A'
        return f'{v*100:+.1f}%' if pct else f'{v:+.2f}'

    hy_z = r.get('hy_z252')
    if hy_z is not None and hy_z == hy_z:
        credit = ('크레딧 공포 구간(역발상 기회)' if hy_z > 1.0 else
                  '크레딧 안일 구간(경계)' if hy_z < -1.0 else '크레딧 중립')
    else:
        credit = '크레딧 N/A'
    return (f'  매크로(전일까지): HY스프레드 z {_f("hy_z252", pct=False)}'
            f' · 20일 {_f("hy_chg20", pct=False)}%p → {credit}'
            f'  |  달러20일 {_f("dxy_ret20")} · 구리20일 {_f("copper_ret20")}'
            f' (참고 — S&P500 선행성은 HY만 유의, macro.py 문서 참조)')


# ── 선행성 검증 ──────────────────────────────────────────────────────────────

def check():
    """S&P500 대비 선행성 검증: IC + 5분위 미래 10일 수익률 (라벨 지평 동일)."""
    spx = _yahoo_close('^GSPC', years=10)
    mac = get_macro_frame(verbose=True)
    if mac is None:
        raise SystemExit('매크로 수집 실패 — 네트워크 확인')
    aligned = (mac.reindex(mac.index.union(spx.index)).ffill()
               .reindex(spx.index).shift(1))
    fwd10 = spx.pct_change(10).shift(-10)

    print(f'\n■ 매크로 신호 → S&P500 미래 10일 수익률 '
          f'({spx.index[0].date()}~, n={len(spx)})')
    for col in MACRO_FEATURES:
        s = aligned[col]
        ok = s.notna() & fwd10.notna()
        if ok.sum() < 200 or s[ok].nunique() < 2:
            print(f'  {col:16s} 데이터 부족 — 스킵')
            continue
        ic = s[ok].corr(fwd10[ok], method='spearman')
        if s[ok].nunique() <= 2:
            m = fwd10[ok].groupby(s[ok]).mean() * 100
            desc = '  '.join(f'{int(k)}={v:+.2f}%' for k, v in m.items())
        else:
            q = pd.qcut(s[ok], 5, labels=False, duplicates='drop')
            m = fwd10[ok].groupby(q).mean() * 100
            desc = '  '.join(f'Q{int(k)+1} {v:+.2f}%' for k, v in m.items())
        print(f'  {col:16s} IC {ic:+.3f}   {desc}')
    print('\n  해석: 미국 시장에서 유의한 선행은 hy_z252 (공포 수준 역발상).')
    print('  달러/구리는 신흥국(KOSPI) 대비 선행성이 크게 약함 — 참고 피처.')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'check'
    if cmd == 'check':
        check()

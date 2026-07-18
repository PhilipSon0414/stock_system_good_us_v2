"""
분석 대상 유니버스 (v2)

v1 대비 변경점:
  - 상장폐지/티커 변경 정리: SGEN(2023 화이자 인수) 제거, FISV→FI 변경
  - 데이터 수집 실패 종목은 daily_scan이 명시적으로 리포트에 출력 (조용한 skip 금지)

⚠ 생존 편향(survivorship bias) 주의:
  이 목록은 "현재 시점에 살아있는" 종목의 하드코딩 목록이다.
  이 목록으로 과거를 백테스트하면 적중률이 부풀려진다.
  backtest 실행 시 이 경고가 함께 출력되며, 백테스트 수치는
  '같은 유니버스 내 상대 비교'로만 해석해야 한다.
"""

_SP500_CORE = [
    # 메가캡
    'AAPL','MSFT','NVDA','AMZN','META','GOOGL','TSLA','BRK-B','JPM','V',
    'UNH','XOM','MA','LLY','JNJ','PG','HD','AVGO','CVX','MRK',
    'ABBV','COST','PEP','KO','WMT','BAC','TMO','CSCO','CRM','ACN',
    'ABT','MCD','DHR','LIN','NKE','ADBE','TXN','NFLX','AMD','QCOM',
    'NEE','PM','RTX','HON','AMGN','ORCL','INTU','SPGI','IBM','CAT',
    # 금융
    'GS','MS','BLK','SCHW','AXP','COF','USB','WFC','C','PNC',
    # 헬스케어
    'ELV','CI','HUM','MCK','CVS','BMY','GILD','VRTX','MRNA','BIIB',
    # 산업/에너지
    'GE','DE','MMM','BA','LMT','NOC','GD','PH','EMR','ETN',
    # 소비재
    'SBUX','TGT','LOW','TJX','ROST','BKNG','MAR','HLT','MGM','WYNN',
    # 반도체
    'MU','INTC','LRCX','KLAC','AMAT','ASML','TSM','MRVL','ON','SWKS',
    # 클라우드/SaaS
    'NOW','SNOW','PLTR','CRWD','PANW','ZS','OKTA','NET','DDOG','S',
    # 핀테크  (SQ→XYZ, FISV→FI 티커 변경 반영)
    'XYZ','PYPL','COIN','HOOD','AFRM','SOFI','UPST','BILL','FIS','FI',
]

_NASDAQ_GROWTH = [
    # AI/반도체 소형
    'SMCI','ARM','IONQ','QUBT','RGTI','QBTS',
    # 바이오/헬스  (SGEN: 2023 인수·상장폐지 → 제거)
    'MRNA','BNTX','NVAX','ACMR','NBIX','ALNY','BMRN',
    # 클린에너지
    'ENPH','SEDG','FSLR','RUN','NOVA','ARRY','STEM','BE',
    # 전기차
    'RIVN','LCID','NIO','XPEV','LI',
    # 비트코인/크립토
    'MSTR','MARA','RIOT','HUT','CLSK','BTBT','CIFR','WULF',
    # 우주/방산
    'RKLB','SPCE','ASTS','LUNR','RDW','KTOS','AVAV',
    # 고성장 소형
    'DUOL','RDDT','CAVA','APP','KVYO','MNDY','GTLB','MDB','ESTC',
    'BRZE','TOST','DOCS','FROG','DV','VRNS','NCNO','JAMF',
]


def get_ticker_list(market: str = 'ALL') -> list[str]:
    """market: 'SP500' | 'GROWTH' | 'ALL'"""
    if market == 'SP500':
        base = _SP500_CORE
    elif market == 'GROWTH':
        base = _NASDAQ_GROWTH
    else:
        base = _SP500_CORE + _NASDAQ_GROWTH
    return list(dict.fromkeys(base))

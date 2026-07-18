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


# 학습 전용 확장 유니버스 — 스캔 대상은 아니지만 학습 데이터 다양성을 높인다.
# (더 많은 종목·섹터의 급등 패턴을 학습해 일반화 성능 개선)
_TRAIN_EXTRA = [
    # 산업/운송
    'UNP','CSX','NSC','FDX','UPS','DAL','UAL','LUV','GM','F',
    'CMI','PCAR','URI','PWR','FAST','GWW','IR','ROK','DOV','AME',
    'CARR','OTIS','JCI','TT','WM','RSG','LHX','HWM','AXON','HEI',
    # 에너지
    'COP','EOG','SLB','HAL','OXY','DVN','FANG','MPC','PSX','VLO',
    'KMI','WMB','LNG','BKR',
    # 소재
    'FCX','NEM','NUE','STLD','DOW','DD','APD','SHW','ECL','ALB',
    'CF','VMC','MLM',
    # 금융/거래소
    'BX','KKR','APO','TROW','STT','BK','TFC','MTB','FITB','KEY',
    'RF','HBAN','CFG','ALL','TRV','PGR','CB','MET','PRU','AFL',
    'AIG','ICE','CME','NDAQ','MCO','MSCI',
    # 헬스케어
    'PFE','REGN','ZTS','ISRG','SYK','BSX','MDT','EW','DXCM','IDXX',
    'IQV','RMD','HCA','PODD','WAT',
    # 테크/반도체
    'ANET','CDNS','SNPS','ADI','NXPI','MCHP','TER','ENTG','MPWR','FTNT',
    'TEAM','WDAY','ADSK','TTD','SHOP','SE','MELI','UBER','ABNB','DASH',
    'ROKU','SPOT','PINS','SNAP','TWLO','U','PATH','AI','SOUN',
    # 통신/미디어/소비재
    'DIS','CMCSA','T','VZ','TMUS','CHWY','DKNG','CZR','LVS','RCL',
    'CCL','NCLH','DPZ','CMG','YUM','QSR','HSY','CL','KMB','EL',
    'ULTA','LULU','DECK','CROX','BURL','DG','DLTR','FIVE',
    # 크립토 채굴 추가
    'HIVE','BITF','IREN',
]


def get_ticker_list(market: str = 'ALL') -> list[str]:
    """market: 'SP500' | 'GROWTH' | 'ALL' | 'TRAIN'
    'TRAIN'은 학습 전용 확장 목록(스캔 유니버스 + 확장 종목)."""
    if market == 'SP500':
        base = _SP500_CORE
    elif market == 'GROWTH':
        base = _NASDAQ_GROWTH
    elif market == 'TRAIN':
        base = _SP500_CORE + _NASDAQ_GROWTH + _TRAIN_EXTRA
    else:
        base = _SP500_CORE + _NASDAQ_GROWTH
    return list(dict.fromkeys(base))

#!/usr/bin/env python3
"""
+10% 급등 예측 모델 — v2

v1의 스테이지 분류 모델을 폐기하고 목표에 직결되는 이진 분류로 재설계:

  라벨:  향후 SURGE_HORIZON(10) 거래일 내 고가가 오늘 종가 대비
         +SURGE_TARGET(10%) 이상 도달하면 1, 아니면 0.
         → v1은 "하루 +4%"의 8-클래스 전조를 예측했는데, 이는 사용자의
           목표(+10% 급등)와 다른 대상이었다.

  v1 문제 해결:
    - 피처는 indicators.MODEL_FEATURES에서 옴 — validate_features()가
      학습·예측 양쪽에서 누락/전체-NaN을 즉시 에러로 잡는다 (조용한 NaN 금지).
    - 학습 데이터 구축이 전 구간 벡터화된 지표를 그대로 사용 →
      VCP/OBV_Diverge의 학습-실전 불일치 해소.
    - Walk-forward 백테스트에 HORIZON만큼 퍼지 갭(purge gap) 적용 —
      라벨의 미래 구간이 학습/테스트에 겹치는 누출 차단.
    - 확률 보정(isotonic): 불균형 데이터에서 predict_proba 과대평가 문제 해결.
      리포트의 "확률"은 보정된 값이다.
    - 임계값은 검증셋에서 목표 정밀도(TARGET_PRECISION)를 만족하는 지점으로
      선택해 모델 파일에 저장 — 정밀도를 출력만 하고 버리지 않는다.

사용:
  python3 surge_model.py train      # 데이터 구축 + walk-forward 평가 + 모델 저장
  python3 surge_model.py backtest   # walk-forward 평가만
"""

import sys
import pickle
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from config import (SURGE_TARGET, SURGE_HORIZON, HISTORY_PERIOD, MIN_PRICE,
                    MIN_AVG_VOLUME, MIN_TRAIN_ROWS, CALIB_FRACTION,
                    TARGET_PRECISION, MIN_PROB_FLOOR)
from indicators import add_all, validate_features, MODEL_FEATURES, TREND_FEATURES
from data_fetcher import get_ohlcv, FetchLog
from universe import get_ticker_list

HERE = Path(__file__).parent
MODEL_PATH = HERE / 'surge_model.pkl'


def feature_columns() -> list[str]:
    cols = list(MODEL_FEATURES)
    for c in TREND_FEATURES:
        cols += [f'd{c}_1d', f'd{c}_3d', f'd{c}_5d']
    return cols


def add_trend_deltas(df: pd.DataFrame) -> pd.DataFrame:
    for c in TREND_FEATURES:
        df[f'd{c}_1d'] = df[c] - df[c].shift(1)
        df[f'd{c}_3d'] = df[c] - df[c].shift(3)
        df[f'd{c}_5d'] = df[c] - df[c].shift(5)
    return df


def make_labels(df: pd.DataFrame) -> pd.Series:
    """향후 HORIZON일 내 고가 기준 +TARGET 도달 여부. 미래 구간이 부족한
    마지막 HORIZON일은 NaN → 학습에서 제외 (look-ahead 누출 방지)."""
    fut_high = df['High'].rolling(SURGE_HORIZON).max().shift(-SURGE_HORIZON)
    fwd_gain = fut_high / df['Close'] - 1
    label = (fwd_gain >= SURGE_TARGET).astype(float)
    label[fwd_gain.isna()] = np.nan
    return label


def build_dataset(tickers: list[str] | None = None,
                  verbose: bool = True) -> pd.DataFrame:
    tickers = tickers or get_ticker_list('ALL')
    log = FetchLog()
    frames = []

    for i, t in enumerate(tickers, 1):
        raw = get_ohlcv(t, period=HISTORY_PERIOD, log=log)
        if len(raw) < 260:
            continue
        df = add_all(raw)
        df = add_trend_deltas(df)
        df['label']  = make_labels(df)
        df['ticker'] = t
        df['date']   = df.index

        # 스캔 대상과 동일한 필터를 학습에도 적용 (분포 일치)
        mask = (df['Close'] >= MIN_PRICE) & (df['VolMA'] >= MIN_AVG_VOLUME)
        df = df[mask & df['label'].notna() & df['VolRatio'].notna()]
        if len(df):
            frames.append(df[feature_columns() + ['label', 'ticker', 'date']])

        if verbose and (i % 20 == 0 or i == len(tickers)):
            n = sum(len(f) for f in frames)
            print(f'  [{i:3d}/{len(tickers)}] rows={n}', end='\r')

    if verbose:
        print()
        for line in log.summary_lines():
            print(line)

    if not frames:
        raise RuntimeError('데이터 구축 실패: 수집된 종목이 없습니다.')
    data = pd.concat(frames, ignore_index=True)
    validate_features(data, feature_columns())
    return data.sort_values('date').reset_index(drop=True)


# ── 학습 + 보정 + 임계값 ─────────────────────────────────────────────────────

def _fit_calibrated(train: pd.DataFrame) -> dict:
    """시간순 앞 (1-CALIB_FRACTION)으로 학습, 뒤 CALIB_FRACTION으로
    isotonic 보정 + 임계값 선택."""
    feat_cols = feature_columns()
    n_cal = max(200, int(len(train) * CALIB_FRACTION))
    fit_part, cal_part = train.iloc[:-n_cal], train.iloc[-n_cal:]

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6,
        min_samples_leaf=30, class_weight='balanced', random_state=42,
    )
    model.fit(fit_part[feat_cols].values, fit_part['label'].values.astype(int))

    p_raw = model.predict_proba(cal_part[feat_cols].values)[:, 1]
    y_cal = cal_part['label'].values.astype(int)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(p_raw, y_cal)
    p_cal = iso.predict(p_raw)

    # 목표 정밀도를 만족하는 최소 임계값 (검증 표본 20개 이상 조건)
    threshold = None
    for t in np.arange(0.10, 0.91, 0.02):
        picked = p_cal >= t
        if picked.sum() >= 20:
            prec = y_cal[picked].mean()
            if prec >= TARGET_PRECISION:
                threshold = float(t)
                break
    if threshold is None:
        threshold = max(MIN_PROB_FLOOR, float(np.quantile(p_cal, 0.95)))

    return {'model': model, 'iso': iso, 'feat_cols': feat_cols,
            'threshold': threshold, 'base_rate': float(train['label'].mean())}


def walk_forward_backtest(data: pd.DataFrame, n_folds: int = 3) -> None:
    print('\n' + '═' * 68)
    print('  Walk-Forward 백테스트 (+10%/10일 이진 분류, 퍼지 갭 적용)')
    print('  ⚠ 유니버스가 현재 생존 종목 목록이므로 생존 편향이 있습니다.')
    print('    절대 수치보다 base rate 대비 향상(lift)으로 해석하세요.')
    print('═' * 68)

    dates = np.sort(data['date'].unique())
    fold_edges = [int(len(dates) * f) for f in (0.5, 0.65, 0.8, 0.95)]

    for k in range(n_folds):
        train_end  = dates[fold_edges[k]]
        test_start = dates[min(fold_edges[k] + SURGE_HORIZON, len(dates) - 1)]
        test_end   = dates[fold_edges[k + 1]]

        train = data[data['date'] <= train_end]
        test  = data[(data['date'] > test_start) & (data['date'] <= test_end)]
        if len(train) < MIN_TRAIN_ROWS or len(test) < 200:
            continue

        fit = _fit_calibrated(train)
        p_raw = fit['model'].predict_proba(test[fit['feat_cols']].values)[:, 1]
        p_cal = fit['iso'].predict(p_raw)
        y = test['label'].values.astype(int)

        base = y.mean()
        picked = p_cal >= fit['threshold']
        prec = y[picked].mean() if picked.sum() else float('nan')
        rec  = (picked & (y == 1)).sum() / max(1, (y == 1).sum())
        top_dec = p_cal >= np.quantile(p_cal, 0.9)
        prec_td = y[top_dec].mean()

        print(f'\n  [Fold {k+1}] train={len(train)}  test={len(test)}'
              f'  (~{pd.Timestamp(test_end).date()})')
        print(f'    base rate(+10% 도달률): {base:.1%}')
        print(f'    임계값 {fit["threshold"]:.2f} 이상 픽: {picked.sum()}건'
              f'  정밀도 {prec:.1%}  재현율 {rec:.1%}'
              f'  lift {prec/base:.2f}x' if picked.sum() else
              f'    임계값 {fit["threshold"]:.2f} 이상 픽 없음')
        print(f'    상위 10% 확률 구간 정밀도: {prec_td:.1%}  lift {prec_td/base:.2f}x')


def train_and_save(data: pd.DataFrame) -> dict:
    if len(data) < MIN_TRAIN_ROWS:
        raise RuntimeError(f'학습 샘플 부족: {len(data)} < {MIN_TRAIN_ROWS}')
    fit = _fit_calibrated(data)
    payload = {**fit,
               'target': SURGE_TARGET, 'horizon': SURGE_HORIZON,
               'trained_at': datetime.now().isoformat(),
               'n_rows': len(data)}
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)
    print(f'\n  모델 저장: {MODEL_PATH}')
    print(f'  샘플 {len(data)}건  base rate {fit["base_rate"]:.1%}'
          f'  임계값 {fit["threshold"]:.2f}')
    return payload


def load_model() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def predict_prob(df: pd.DataFrame, loaded: dict | None = None) -> dict | None:
    """add_all() 적용된 DataFrame의 마지막 행에 대해 보정 확률 반환.
    모델이 없으면 None — 호출자가 명시적으로 '모델 없음'을 표시해야 한다."""
    loaded = loaded or load_model()
    if loaded is None:
        return None
    df = add_trend_deltas(df.copy())
    validate_features(df, loaded['feat_cols'])
    x = df[loaded['feat_cols']].iloc[[-1]].values
    p_raw = loaded['model'].predict_proba(x)[0, 1]
    p_cal = float(loaded['iso'].predict([p_raw])[0])
    return {
        'prob':      round(p_cal, 3),
        'threshold': loaded['threshold'],
        'passes':    p_cal >= loaded['threshold'],
        'base_rate': loaded.get('base_rate'),
    }


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'train'
    print(f'\n  +10%/{SURGE_HORIZON}일 급등 예측 모델 — {cmd}')
    print('  [1] 데이터 구축...')
    data = build_dataset()
    pos = data['label'].mean()
    print(f'  전체 {len(data)}건  양성(+10% 도달) 비율 {pos:.1%}')

    print('  [2] Walk-forward 백테스트...')
    walk_forward_backtest(data)

    if cmd == 'train':
        print('\n  [3] 전체 데이터로 최종 학습...')
        train_and_save(data)


if __name__ == '__main__':
    main()

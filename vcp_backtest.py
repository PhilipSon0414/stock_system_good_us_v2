"""VCP 돌파 전략 vs 기존 방식(급등확률 모델 + 기술점수 + 엘리트 규칙) 비교 백테스트.

  python vcp_backtest.py [--cache PATH] [--refresh] [--step 63]

같은 종목·같은 기간·같은 비용 가정에서 두 방식을 비교한다.

기존 방식은 과거 시점 추천 기록이 2026-10-29 이후뿐이라, 현재 코드로 walk-forward
재구성한다(당시 실제 추천이 아니다):
  - 분기마다 직전 2년(실전 학습 구간과 동일한 유효 행)으로 모델을 다시 학습하고,
    라벨 구간을 purge한 뒤 다음 분기를 표본 외로 예측한다.
  - 게이트: 보정확률 ≥ 모델 임계값, 5일 급등 15% 미만, 기술점수 ≥ TECH_SCORE_GATE,
    합산 ≥ FINAL_MIN_SCORE.  엘리트: 여기에 엘리트 제외 규칙(ATR·볼린저·20일 고점·MA200).
  - 재구성할 수 없어 뺀 것: 실적 발표 블랙아웃, 재무·공매도 가점, tracker 실측 게이트 조정.
청산은 execution.simulate(다음 시가 진입, +12% / 3×ATR 손절 / D+3 컷 / 20거래일).

VCP는 vcp_strategy 참고. 신호(진입 조건)와 청산 규칙의 효과를 분리하려고
신호 × 청산을 교차한 조합도 함께 낸다. 종목별로 보유 중에는 새 진입을 받지 않는다.
유니버스가 현재 생존 종목이라 두 방식 모두 생존 편향이 있다 — 절대 수치보다 상대 비교용.
"""
from __future__ import annotations

import argparse
import json
import pickle
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import vcp_strategy as vcp
from config import (SURGE_HORIZON, MIN_PRICE, MIN_AVG_VOLUME, TECH_SCORE_GATE, FINAL_MIN_SCORE,
                    W_SCORE, W_MODEL, MIN_TRAIN_ROWS, ELITE_MAX_ATR, ELITE_MAX_BB_EXPANSION,
                    ELITE_MAX_DROP_20D_HIGH, ELITE_MAX_ABOVE_MA200)
from data_fetcher import get_ohlcv, FetchLog
from execution import simulate as execute, policy
from indicators import add_all
from order_block import get_order_blocks
from scorer import score_technical
from surge_model import add_trend_deltas, make_labels, feature_columns, _fit_calibrated
from universe import get_ticker_list

BT_PERIOD = '10y'
TRAIN_DAYS = 504          # 실전 HISTORY_PERIOD=3y에서 지표 워밍업을 뺀 유효 학습 구간
DEFAULT_CACHE = Path(tempfile.gettempdir()) / 'vcp_bt_cache.pkl'
OUT_JSON = Path(__file__).parent / 'reports' / 'vcp_comparison.json'


def load_data(cache: Path, refresh: bool = False) -> tuple[dict, pd.Series]:
    if cache.exists() and not refresh:
        return pickle.loads(cache.read_bytes())
    log, out = FetchLog(), {}
    tickers = get_ticker_list('TRAIN')
    for i, t in enumerate(tickers, 1):
        raw = get_ohlcv(t, period=BT_PERIOD, log=log)
        if len(raw) >= 260:
            out[t] = raw
        if i % 25 == 0:
            print(f'  fetch {i}/{len(tickers)}', end='\r')
    spy = get_ohlcv('SPY', period=BT_PERIOD, log=log)['Close']
    cache.write_bytes(pickle.dumps((out, spy)))
    return out, spy


# ── 기존 방식 walk-forward 재구성 ────────────────────────────────────────────

def build_frames(data: dict) -> tuple[dict, pd.DataFrame]:
    frames, rows = {}, []
    cols = feature_columns()
    for t, raw in data.items():
        df = add_trend_deltas(add_all(raw.copy()))
        df['label'] = make_labels(df)
        df['pos'] = np.arange(len(df))
        frames[t] = df
        ok = (df['Close'] >= MIN_PRICE) & (df['VolMA'] >= MIN_AVG_VOLUME) & df['VolRatio'].notna()
        part = df.loc[ok, cols + ['label', 'pos']].copy()
        part['ticker'], part['date'] = t, part.index
        rows.append(part)
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return frames, panel.sort_values('date').reset_index(drop=True)


def walk_forward_probs(panel: pd.DataFrame, step: int) -> tuple[pd.DataFrame, list[dict]]:
    """표본 외 보정확률. 각 폴드는 직전 TRAIN_DAYS 거래일로 학습(라벨 구간 purge)."""
    dates = np.sort(panel['date'].unique())
    first = 250 + TRAIN_DAYS + SURGE_HORIZON
    out, folds = [], []
    for k in range(first, len(dates), step):
        train_end = dates[k - SURGE_HORIZON - 1]
        train_start = dates[max(0, k - SURGE_HORIZON - 1 - TRAIN_DAYS)]
        train = panel[(panel.date >= train_start) & (panel.date <= train_end) & panel.label.notna()]
        test = panel[(panel.date >= dates[k]) & (panel.date <= dates[min(k + step, len(dates)) - 1])]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue
        fit = _fit_calibrated(train)
        p = fit['iso'].predict(fit['model'].predict_proba(test[fit['feat_cols']].values)[:, 1])
        part = test[['ticker', 'date', 'pos', 'label']].copy()
        part['prob'], part['thr'] = p, fit['threshold']
        out.append(part)
        picked = part[part.prob >= fit['threshold']]
        folds.append({'test_start': str(pd.Timestamp(dates[k]).date()), 'threshold': round(fit['threshold'], 2),
                      'n_test': len(part), 'n_pass': len(picked),
                      'base_rate': round(float(part.label.mean()), 3),
                      'precision': round(float(picked.label.mean()), 3) if len(picked) else None})
        print(f"  fold {folds[-1]['test_start']} thr={fit['threshold']:.2f} pass={len(picked)}"
              f" prec={folds[-1]['precision']} base={folds[-1]['base_rate']}")
    return pd.concat(out, ignore_index=True), folds


class Existing:
    """재구성한 기존 방식 신호. 기술점수는 비용이 커서 필요한 행만 계산한다."""

    def __init__(self, frames: dict, probs: pd.DataFrame):
        self.frames, self._tech = frames, {}
        self.by_ticker = {t: g.set_index('pos') for t, g in probs[probs.prob >= probs.thr].groupby('ticker')}

    def tech(self, t: str, i: int) -> int:
        if (t, i) not in self._tech:
            df = self.frames[t].iloc[:i + 1]
            self._tech[(t, i)] = score_technical(df, get_order_blocks(df), info=None)[0]
        return self._tech[(t, i)]

    def candidates(self, t: str, elite: bool) -> list[tuple[int, dict]]:
        if t not in self.by_ticker:
            return []
        df, g = self.frames[t], self.by_ticker[t]
        idx = g.index.to_numpy()
        ok = (df['Gain5D'].to_numpy()[idx] < 0.15)
        if elite:
            close = df['Close'].to_numpy()
            high20 = df['High'].rolling(20).max().to_numpy()
            ok &= ~(df['ATRRel'].to_numpy()[idx] > ELITE_MAX_ATR)
            ok &= ~(df['BBCompress'].to_numpy()[idx] >= ELITE_MAX_BB_EXPANSION)
            ok &= ~(close[idx] / high20[idx] - 1 <= -ELITE_MAX_DROP_20D_HIGH)
            ok &= ~(close[idx] / df['MA200'].to_numpy()[idx] - 1 > ELITE_MAX_ABOVE_MA200)
        return [(int(i), {'prob': float(g.at[i, 'prob'])}) for i in idx[ok]]

    def accept(self, t: str, i: int, meta: dict) -> bool:
        tech = self.tech(t, i)
        return tech >= TECH_SCORE_GATE and round(W_SCORE * tech + W_MODEL * meta['prob'] * 100) >= FINAL_MIN_SCORE


# ── 청산 엔진 ────────────────────────────────────────────────────────────────

def exit_existing(df: pd.DataFrame, i: int, meta: dict) -> dict | None:
    ev = execute(df.iloc[i + 1:i + 1 + SURGE_HORIZON], SURGE_HORIZON, policy(df['ATRRel'].iat[i]))
    if ev is None:
        return None
    hold = int(np.searchsorted(df.index.values, np.datetime64(ev['exit_date']))) - i
    return {**ev, 'hold_days': hold}


def make_exit_vcp(params: dict):
    def run(df: pd.DataFrame, i: int, meta: dict) -> dict | None:
        return vcp.simulate_exit(df, i, params, meta.get('pivot'))
    return run


def run_trades(tickers, frames, candidates, accept, exit_fn, start) -> list[dict]:
    """종목별 시간순. 보유 중(청산일까지)에는 새 신호를 받지 않는다."""
    trades = []
    for t in tickers:
        if t not in frames:
            continue
        df, busy_until = frames[t], -1
        for i, meta in candidates(t):
            if i <= busy_until or df.index[i] < start:
                continue
            if accept is not None and not accept(t, i, meta):
                continue
            ev = exit_fn(df, i, meta)
            if ev is None or 'skipped' in ev:
                continue
            busy_until = i + ev['hold_days']
            trades.append({'ticker': t, 'signal_date': str(df.index[i].date()), **ev})
    return trades


# ── 집계 ─────────────────────────────────────────────────────────────────────

def stats(trades: list[dict]) -> dict:
    if not trades:
        return {'n': 0}
    r = np.array([t['rule_return'] for t in trades])
    wins, losses = r[r > 0], r[r <= 0]
    hold = np.array([t['hold_days'] for t in trades])
    outcomes = pd.Series([t['rule_outcome'] for t in trades]).value_counts(normalize=True)
    return {'n': len(r), 'win_rate': round(float((r > 0).mean()), 4),
            'avg_return': round(float(r.mean()), 5), 'median_return': round(float(np.median(r)), 5),
            'avg_win': round(float(wins.mean()), 5) if len(wins) else None,
            'avg_loss': round(float(losses.mean()), 5) if len(losses) else None,
            'payoff': round(float(wins.mean() / -losses.mean()), 2) if len(wins) and len(losses) and losses.mean() < 0 else None,
            'profit_factor': round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() < 0 else None,
            'avg_hold_days': round(float(hold.mean()), 1),
            'return_per_day': round(float(r.sum() / hold.sum()), 6),
            'worst': round(float(r.min()), 4), 'best': round(float(r.max()), 4),
            'outcomes': {k: round(float(v), 3) for k, v in outcomes.items()}}


def by_year(trades: list[dict]) -> dict:
    if not trades:
        return {}
    d = pd.DataFrame(trades)
    d['year'] = d['entry_date'].str[:4]
    return {y: {'n': len(g), 'win_rate': round(float((g.rule_return > 0).mean()), 3),
                'avg_return': round(float(g.rule_return.mean()), 4)} for y, g in d.groupby('year')}


def by_regime(trades: list[dict], mkt: pd.Series) -> dict:
    out = {}
    for name, flag in (('spy_above_ma50_200', True), ('spy_below', False)):
        sub = [t for t in trades if bool(mkt.get(pd.Timestamp(t['signal_date']), False)) == flag]
        s = stats(sub)
        out[name] = {k: s.get(k) for k in ('n', 'win_rate', 'avg_return', 'payoff')}
    return out


def boot_diff(a: list[dict], b: list[dict], n_boot: int = 2000, seed: int = 7) -> dict | None:
    """진입 월 블록 부트스트랩으로 (a - b)의 승률·평균수익 차이 90% 신뢰구간."""
    if not a or not b:
        return None
    fa, fb = (pd.DataFrame(x).assign(m=lambda d: d.entry_date.str[:7]) for x in (a, b))
    months = sorted(set(fa.m) | set(fb.m))
    ga = {m: g.rule_return.to_numpy() for m, g in fa.groupby('m')}
    gb = {m: g.rule_return.to_numpy() for m, g in fb.groupby('m')}
    rng, dw, dm = np.random.default_rng(seed), [], []
    empty = np.array([])
    for _ in range(n_boot):
        pick = rng.choice(months, len(months))
        ra = np.concatenate([ga.get(m, empty) for m in pick])
        rb = np.concatenate([gb.get(m, empty) for m in pick])
        if len(ra) and len(rb):
            dw.append((ra > 0).mean() - (rb > 0).mean()); dm.append(ra.mean() - rb.mean())
    q = lambda x: [round(float(v), 4) for v in np.quantile(x, [0.05, 0.5, 0.95])]
    return {'win_rate_diff_ci90': q(dw), 'avg_return_diff_ci90': q(dm)}


def fmt(name: str, s: dict) -> str:
    if not s.get('n'):
        return f'{name:<34} n=0'
    pct = lambda v: '   n/a' if v is None else f'{v * 100:+6.2f}%'
    return (f"{name:<34} n={s['n']:5d} 승률 {s['win_rate'] * 100:5.1f}% 평균 {pct(s['avg_return'])} "
            f"중앙 {pct(s['median_return'])} 평균익 {pct(s['avg_win'])} 평균손 {pct(s['avg_loss'])} "
            f"손익비 {s['payoff']} PF {s['profit_factor']} 보유 {s['avg_hold_days']:5.1f}일")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--step', type=int, default=63, help='walk-forward 재학습 간격(거래일)')
    args = ap.parse_args()

    data, spy = load_data(args.cache, args.refresh)
    scan_universe = [t for t in get_ticker_list('ALL') if t in data]
    print(f'종목 {len(data)}개(매매 대상 {len(scan_universe)}개), 기간 {BT_PERIOD}')

    frames, panel = build_frames(data)
    probs, folds = walk_forward_probs(panel, args.step)
    start = probs['date'].min()
    print(f'표본 외 평가 구간: {start.date()} ~ {probs["date"].max().date()}')

    mkt = vcp.market_ok(spy)
    closes = pd.DataFrame({t: df['Close'] for t, df in data.items()})
    rs = vcp.rs_percentile(closes)
    vframes = {t: vcp.add_features(raw) for t, raw in data.items()}

    existing = Existing(frames, probs)
    ex_gate = lambda t: existing.candidates(t, elite=False)
    ex_elite = lambda t: existing.candidates(t, elite=True)

    def vcp_signals(params, use_market=True):
        cache = {}
        def cand(t):
            if t not in cache:
                cache[t] = [(s['i'], s) for s in vcp.find_signals(
                    vframes[t], rs[t], mkt if use_market else None, params)]
            return cache[t]
        return cand

    P = vcp.PARAMS
    variants = {
        'VCP 기본': (vcp_signals(P), P),
        'VCP 시장필터 없음': (vcp_signals(P, use_market=False), P),
        'VCP 엄격(신고가-20%·고갈0.5·RS90)': (vcp_signals({**P, 'max_dist_high': 0.20, 'dryup_max': 0.5, 'rs_min': 0.9}), P),
        'VCP 완화(RS70·고갈0.8)': (vcp_signals({**P, 'rs_min': 0.7, 'dryup_max': 0.8}), P),
        'VCP 고갈조건 없음': (vcp_signals({**P, 'dryup_max': 99.0}), P),
        'VCP 고갈조건 없음·RS70': (vcp_signals({**P, 'dryup_max': 99.0, 'rs_min': 0.7}), P),
        'VCP 손절5%': (vcp_signals(P), {**P, 'stop_pct': 0.05}),
        'VCP 20일선 처음부터 적용': (vcp_signals(P), {**P, 'trail_from_start': True}),
        'VCP 익절20%': (vcp_signals(P), {**P, 'partial_at': 0.20}),
        'VCP 10일선 트레일': (vcp_signals(P), {**P, 'trail_ma': 10}),
    }

    results = {}
    def record(name, trades):
        results[name] = trades
        print(fmt(name, stats(trades)))

    print('\n── 기존 방식 (walk-forward 재구성) ──')
    record('기존 게이트 통과 / 기존 청산', run_trades(scan_universe, frames, ex_gate, existing.accept, exit_existing, start))
    record('기존 엘리트 / 기존 청산', run_trades(scan_universe, frames, ex_elite, existing.accept, exit_existing, start))
    print('\n── VCP ──')
    for name, (cand, params) in variants.items():
        record(name, run_trades(scan_universe, vframes, cand, None, make_exit_vcp(params), start))
    print('\n── 교차 (신호와 청산 효과 분리) ──')
    vexit = lambda df, i, meta: vcp.simulate_exit(df, i, P, None)
    record('기존 엘리트 신호 / VCP 청산', run_trades(scan_universe, frames, ex_elite, existing.accept, vexit, start))
    record('기존 게이트 신호 / VCP 청산', run_trades(scan_universe, frames, ex_gate, existing.accept, vexit, start))
    atr_frames = {t: df.assign(ATRRel=df['AtrRel']) for t, df in vframes.items()}
    record('VCP 신호 / 기존 청산', run_trades(scan_universe, atr_frames, variants['VCP 기본'][0], None, exit_existing, start))

    def in_market(trades):
        return [t for t in trades if bool(mkt.get(pd.Timestamp(t['signal_date']), False))]
    print('\n── 같은 시장 조건(SPY > 50·200일선)에서만 ──')
    record('기존 엘리트 / 기존 청산 (시장필터)', in_market(results['기존 엘리트 / 기존 청산']))
    record('기존 게이트 / 기존 청산 (시장필터)', in_market(results['기존 게이트 통과 / 기존 청산']))

    base, ref = results['VCP 기본'], results['기존 엘리트 / 기존 청산']
    diffs = {'VCP 기본 - 기존 엘리트': boot_diff(base, ref),
             'VCP 기본 - 기존 게이트': boot_diff(base, results['기존 게이트 통과 / 기존 청산']),
             'VCP 기본 - 기존 엘리트(시장필터)': boot_diff(base, results['기존 엘리트 / 기존 청산 (시장필터)'])}
    print('\n── 차이의 90% 신뢰구간 (월 블록 부트스트랩) ──')
    for k, v in diffs.items():
        print(f'  {k}: {v}')

    recent_cut = '2026-10-29'
    recent_vcp = [t for t in base if t['signal_date'] >= recent_cut]
    open_vcp = []
    for t in scan_universe:
        for i, s in variants['VCP 기본'][0](t):
            if str(s['date'].date()) >= recent_cut:
                open_vcp.append({'ticker': t, 'date': str(s['date'].date()), 'pivot': round(s['pivot'], 2),
                                 'close': round(s['close'], 2), 'depths': s['depths']})
    print(f'\n실전 기록 구간({recent_cut}~) VCP 신호 {len(open_vcp)}건, 청산 완료 {len(recent_vcp)}건')

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        'generated_for': str(spy.index[-1].date()), 'period': BT_PERIOD,
        'oos_start': str(start.date()), 'n_tickers': len(scan_universe),
        'vcp_version': vcp.VERSION, 'vcp_params': P, 'walk_forward_step': args.step,
        'folds': folds,
        'summary': {k: stats(v) for k, v in results.items()},
        'by_year': {k: by_year(results[k]) for k in ('VCP 기본', '기존 엘리트 / 기존 청산', '기존 게이트 통과 / 기존 청산')},
        'by_regime': {k: by_regime(results[k], mkt) for k in ('VCP 시장필터 없음', '기존 엘리트 / 기존 청산', '기존 게이트 통과 / 기존 청산')},
        'diff_ci90': diffs, 'recent_vcp_signals': open_vcp,
        'trades': {k: results[k] for k in ('VCP 기본', '기존 엘리트 / 기존 청산')},
        'caveats': ['survivorship-biased universe', 'existing method is a walk-forward reconstruction, not live picks',
                    'no point-in-time fundamentals or earnings dates'],
    }, ensure_ascii=False, indent=1, default=str))
    print(f'저장: {OUT_JSON}')


if __name__ == '__main__':
    main()

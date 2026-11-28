import unittest
import numpy as np
import pandas as pd
import vcp_strategy as vcp

P0 = {**vcp.PARAMS, 'fee_bps': 0, 'slippage_bps': 0}


def frame(closes, opens=None, highs=None, lows=None, volume=1e6):
    c = np.asarray(closes, dtype=float)
    o = np.asarray(opens, dtype=float) if opens is not None else c.copy()
    h = np.asarray(highs, dtype=float) if highs is not None else np.maximum(o, c)
    l = np.asarray(lows, dtype=float) if lows is not None else np.minimum(o, c)
    return pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c, 'Volume': volume},
                        index=pd.bdate_range('2024-01-01', periods=len(c)))


class Contractions(unittest.TestCase):
    def test_detects_shrinking_pullbacks(self):
        path = [100, 85, 99, 92, 100, 97, 100.5]
        legs = vcp.contractions(np.array(path, float), np.array(path, float), 0.03)
        depths = [round(1 - t / p, 2) for _, p, t in legs]
        self.assertEqual(depths, [0.15, 0.07, 0.03])

    def test_ignores_moves_below_threshold(self):
        path = [100, 99, 100, 98.5, 100]
        self.assertEqual(vcp.contractions(np.array(path, float), np.array(path, float), 0.03), [])


class ExitRules(unittest.TestCase):
    def test_stop_at_seven_percent(self):
        df = frame([100, 100, 99, 94, 94], opens=[100, 100, 99, 99, 94], lows=[100, 100, 99, 90, 94])
        ev = vcp.simulate_exit(df, 0, P0)
        self.assertEqual(ev['rule_outcome'], 'stop')
        self.assertAlmostEqual(ev['rule_return'], -0.07, places=6)

    def test_gap_down_fills_at_open(self):
        df = frame([100, 100, 85], opens=[100, 100, 85])
        ev = vcp.simulate_exit(df, 0, P0)
        self.assertAlmostEqual(ev['rule_return'], -0.15, places=6)

    def test_partial_then_breakeven(self):
        df = frame([100, 100, 116, 118, 100, 99], lows=[100, 100, 100, 116, 99, 99])
        ev = vcp.simulate_exit(df, 0, {**P0, 'trail_ma': 3})
        self.assertEqual(ev['rule_outcome'], 'breakeven')
        self.assertAlmostEqual(ev['rule_return'], 0.5 * 0.15, places=6)

    def test_partial_then_trail_exit_next_open(self):
        closes = [100, 100, 116, 120, 124, 110, 112]
        df = frame(closes, lows=[100, 100, 100, 116, 120, 110, 110])
        ev = vcp.simulate_exit(df, 0, {**P0, 'trail_ma': 3})
        self.assertEqual(ev['rule_outcome'], 'trail')
        self.assertEqual(ev['exit_date'], str(df.index[6].date()))
        self.assertAlmostEqual(ev['rule_return'], 0.5 * 0.15 + 0.5 * 0.12, places=6)

    def test_no_chase_and_unfinished(self):
        df = frame([100, 110, 111], opens=[100, 110, 111])
        self.assertEqual(vcp.simulate_exit(df, 0, P0, pivot=100)['skipped'], 'gap_above_chase_limit')
        self.assertIsNone(vcp.simulate_exit(frame([100, 100, 101]), 0, P0))


class Signals(unittest.TestCase):
    def synthetic(self):
        rng = np.random.default_rng(0)
        up = np.linspace(50, 100, 300)
        base = [100, 97, 93, 89, 86, 85, 88, 92, 95, 98, 99, 97, 95, 93, 92, 94, 96, 98, 99.5,
                98.5, 97.5, 96.5, 97, 98, 98.5, 99, 99.2, 99.3, 99.4, 99.5]
        closes = np.concatenate([up, base, [102.5, 103, 104]])
        vol = np.full(len(closes), 1e6) * (1 + 0.05 * rng.standard_normal(len(closes)))
        vol[-13:-3] = 4e5
        vol[-3] = 2.5e6
        return frame(closes, volume=vol)

    def test_breakout_signal_and_filters(self):
        df = vcp.add_features(self.synthetic())
        rs = pd.Series(0.95, index=df.index)
        sig = vcp.find_signals(df, rs, None)
        self.assertEqual(len(sig), 1)
        s = sig[0]
        self.assertEqual(s['i'], len(df) - 3)
        self.assertGreaterEqual(s['n_contractions'], 2)
        self.assertLess(s['depths'][-1], s['depths'][0])
        self.assertLess(s['pivot'], s['close'])
        self.assertEqual(vcp.find_signals(df, rs * 0.5, None), [])
        self.assertEqual(vcp.find_signals(df, rs, pd.Series(False, index=df.index)), [])

    def test_no_signal_without_volume(self):
        raw = self.synthetic()
        raw.iloc[-3, raw.columns.get_loc('Volume')] = 1e6
        self.assertEqual(vcp.find_signals(vcp.add_features(raw), pd.Series(0.95, index=raw.index), None), [])

    def test_rs_percentile_ranks_cross_section(self):
        idx = pd.bdate_range('2023-01-01', periods=300)
        closes = pd.DataFrame({'A': np.linspace(10, 30, 300), 'B': np.linspace(10, 11, 300),
                               'C': np.linspace(10, 8, 300)}, index=idx)
        last = vcp.rs_percentile(closes).iloc[-1]
        self.assertGreater(last['A'], last['B'])
        self.assertGreater(last['B'], last['C'])


if __name__ == '__main__':
    unittest.main()

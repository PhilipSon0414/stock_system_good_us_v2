import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import position_monitor as pm

RULE = {**pm.RULE, 'trail_ma': 3, 'fee_bps': 0, 'slippage_bps': 0}


def frame(closes, opens=None, highs=None, lows=None):
    c = np.asarray(closes, dtype=float)
    o = np.asarray(opens, dtype=float) if opens is not None else c.copy()
    h = np.asarray(highs, dtype=float) if highs is not None else np.maximum(o, c)
    l = np.asarray(lows, dtype=float) if lows is not None else np.minimum(o, c)
    return pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c, 'Volume': 1e6},
                        index=pd.bdate_range('2026-01-05', periods=len(c)))


class Status(unittest.TestCase):
    def test_hold_shows_standing_orders(self):
        df = frame([100, 101, 103])
        s = pm.status(df, '2026-01-05', 100, from_open=False, rule=RULE)
        self.assertEqual(s['action'], 'hold')
        self.assertEqual((s['stop'], s['partial_target'], s['days_held']), (93.0, 115.0, 3))
        self.assertIn('93.00', s['orders'])

    def test_partial_today_asks_to_raise_stop(self):
        df = frame([100, 105, 116])
        s = pm.status(df, '2026-01-05', 100, from_open=False, rule=RULE)
        self.assertEqual(s['action'], 'raise_stop')
        self.assertEqual((s['stop'], s['partial_done'], s['partial_date']), (100.0, True, '2026-01-07'))
        self.assertAlmostEqual(s['return_pct'], 0.5 * 15 + 0.5 * 16)

    def test_close_below_ma_after_partial_sells_next_open(self):
        df = frame([100, 116, 120, 124, 110], lows=[100, 100, 116, 120, 110])
        s = pm.status(df, '2026-01-05', 100, from_open=False, rule=RULE)
        self.assertEqual(s['action'], 'sell_next_open')
        self.assertIn('시가 전량 매도', s['orders'])

    def test_below_ma_before_partial_is_still_hold(self):
        df = frame([100, 104, 105, 99])
        self.assertEqual(pm.status(df, '2026-01-05', 100, from_open=False, rule=RULE)['action'], 'hold')

    def test_stop_hit_is_closed(self):
        df = frame([100, 99, 94], opens=[100, 99, 97], lows=[100, 99, 92])
        s = pm.status(df, '2026-01-05', 100, from_open=False, rule=RULE)
        self.assertEqual((s['action'], s['exit_reason'], s['exit_date']), ('closed', 'stop', '2026-01-07'))
        self.assertAlmostEqual(s['return_pct'], -7.0)

    def test_entry_day_range_ignored_for_real_fills(self):
        df = frame([100, 101], lows=[90, 100])
        self.assertEqual(pm.status(df, '2026-01-05', 100, from_open=False, rule=RULE)['action'], 'hold')
        self.assertEqual(pm.status(df, '2026-01-05', 100, from_open=True, rule=RULE)['action'], 'closed')

    def test_future_entry_is_pending(self):
        self.assertEqual(pm.status(frame([100]), '2026-02-02', 100, from_open=False)['action'], 'pending_entry')


class Sources(unittest.TestCase):
    def test_holdings_file_env_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'state.json'
            self.assertEqual(pm.load_holdings(p)[0], [])
            p.write_text(json.dumps({'as_of_session': '2026-01-02', 'positions': [
                {'ticker': 'aaa', 'weight': .1, 'stop_pct': .07, 'entry_date': '2026-01-05', 'entry_price': 50},
                {'ticker': 'BBB', 'weight': .1, 'stop_pct': .07}]}))
            rows, note = pm.load_holdings(p)
            self.assertEqual([(r['ticker'], r['entry_price']) for r in rows], [('AAA', 50.0)])
            self.assertIsNone(note)
            p.write_text('{"positions": [{"ticker": "A", "entry_date": "2026-01-05", "entry_price": -1}]}')
            self.assertIn('형식 오류', pm.load_holdings(p)[1])
            with patch.dict(os.environ, {'PORTFOLIO_POSITIONS': '[{"ticker":"ZZZ","entry_date":"2026-01-06","entry_price":10}]'}):
                self.assertEqual(pm.load_holdings(p)[0][0]['ticker'], 'ZZZ')

    def test_recommendations_skip_repeat_while_open_and_summarize(self):
        df = frame([100, 100, 101, 102, 103, 104])
        recs = [{'scan_date': '2026-01-05', 'ticker': 'AAA', 'close': 100, 'elite': True},
                {'scan_date': '2026-01-07', 'ticker': 'AAA', 'close': 101, 'elite': True},
                {'scan_date': '2026-01-12', 'ticker': 'BBB', 'close': 104, 'elite': True}]
        with patch.object(pm, '_elite_history', return_value=recs), \
             patch.object(pm, '_bars', return_value=df):
            out = pm.track_recommendations('2026-01-12')
        self.assertEqual([(r['ticker'], r['action']) for r in out['open']],
                         [('AAA', 'hold'), ('BBB', 'pending_entry')])
        self.assertEqual(out['open'][0]['entry_date'], '2026-01-06')
        self.assertIsNone(pm.summarize_closed([]))
        s = pm.summarize_closed([{'return_pct': 10, 'exit_reason': 'trail', 'days_held': 10},
                                 {'return_pct': -7, 'exit_reason': 'stop', 'days_held': 4}])
        self.assertEqual((s['n'], s['win_rate'], s['avg_return']), (2, .5, .015))


if __name__ == '__main__':
    unittest.main()

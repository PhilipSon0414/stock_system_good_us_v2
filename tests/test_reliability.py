import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import execution
import tracker
import daily_scan
import surge_model
from validation import purged_split
from market_clock import completed_session, session_offset, session_dates, completed_bars
from portfolio_state import load_state
from strategy_compare import compare


def bars(o=100, h=101, l=99, c=100, n=20):
    return pd.DataFrame({'Open':o, 'High':h, 'Low':l, 'Close':c, 'Volume':1e6},
                        index=session_dates('2026-01-05','2026-03-31')[:n])

def rules():
    return {**execution.policy(.05), 'slippage_bps':0, 'fee_bps':0}

class Reliability(unittest.TestCase):
    def test_calendar_dst_and_early_close(self):
        self.assertEqual(completed_session('2026-10-27T07:00+09:00'),'2026-10-26')
        self.assertEqual(completed_session('2026-01-07T06:10+09:00'),'2026-01-05')
        self.assertEqual(completed_session('2026-11-27T18:31Z'),'2026-11-27')
        self.assertEqual(session_offset('2026-11-25',1),'2026-11-27')
    def test_stale_and_partial_bar_rejected(self):
        with self.assertRaises(ValueError): completed_bars(bars(n=1),'2026-01-06')
        df=bars(n=1);df.iloc[0,0]=np.nan
        with self.assertRaises(ValueError): completed_bars(df,'2026-01-05')
    def test_gap_up_does_not_create_fake_profit(self):
        ev=execution.simulate(bars(112,113,110,112),20,rules())
        self.assertEqual(ev['rule_return'],0)
        self.assertEqual(ev['entry_price'],112)
    def test_gap_stop_fills_at_open(self):
        df=bars();df.iloc[1,:4]=[70,75,65,72]
        ev=execution.simulate(df,20,rules())
        self.assertAlmostEqual(ev['rule_return'],-.30)
    def test_same_bar_stop_first(self):
        df=bars();df.iloc[1,:4]=[100,120,80,100]
        self.assertEqual(execution.simulate(df,20,rules())['rule_outcome'],'stop')
    def test_d3_exits_next_open(self):
        df=bars();df.iloc[2,:4]=[95,96,90,91];df.iloc[3,:4]=[92,94,91,93]
        ev=execution.simulate(df,20,rules())
        self.assertEqual(ev['rule_outcome'],'d3')
        self.assertAlmostEqual(ev['rule_return'],-.08)
    def test_missing_bars_and_costs(self):
        df=bars();df.iloc[4,0]=np.nan
        self.assertIsNone(execution.simulate(df,20,rules()))
        self.assertLess(execution.simulate(bars(),20,execution.policy(.05))['rule_return'],0)
    def test_split_has_no_shared_dates_or_overlapping_labels(self):
        dates=session_dates('2024-01-02','2026-01-02')
        data=pd.DataFrame({'date':np.repeat(dates,3)})
        fit,cal=purged_split(data,.8,20)
        self.assertLess(session_offset(fit.date.max(),20),str(cal.date.min().date()))
        self.assertFalse(set(fit.date)&set(cal.date))
    def test_immutable_predictions_and_policy(self):
        with tempfile.TemporaryDirectory() as d, patch.object(tracker,'PRED_DIR',Path(d)):
            rows=[{'ticker':'ABC','close':100,'atr_rel':.05,'prob':.6}]
            self.assertEqual(tracker.save_predictions(rows,'2026-01-02'),1)
            file=next(Path(d).glob('*.jsonl'));original=file.read_text()
            self.assertEqual(tracker.save_predictions([],'2026-01-02'),0)
            self.assertEqual(file.read_text(),original)
            rec=json.loads(original)
            with patch.object(execution,'COACH_TARGET',.20):
                self.assertEqual(tracker._rule_params(rec)[0],.12)
    def test_old_model_is_not_used(self):
        with patch.object(surge_model.pickle,'load',return_value={'horizon':20}):
            self.assertIsNone(surge_model.load_model())
    def test_holdings_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.json'
            self.assertIsNone(load_state('2026-01-02',p)[0])
            p.write_text(json.dumps({'as_of_session':'2026-01-02','positions':[]}))
            self.assertIsNotNone(load_state('2026-01-02',p)[0])
            self.assertIsNone(load_state('2026-01-05',p)[0])
    def test_account_caps_and_absent_state(self):
        candidate={'ticker':'NEW','name':'New','price':100,'atr_rel':.05,'df':bars(n=60)}
        self.assertEqual(daily_scan.build_portfolio([candidate],'normal',None)['positions'],[])
        state={'as_of_session':'2026-01-05','positions':[{'ticker':'HELD','weight':.39,'stop_pct':.15}]}
        with patch.object(daily_scan,'get_ohlcv',return_value=bars(n=1)), patch.object(daily_scan,'_avg_pairwise_corr',return_value=.1):
            out=daily_scan.build_portfolio([candidate],'normal',None,state)
        self.assertLessEqual(out['total_deploy'],.4+1e-9)
        self.assertLessEqual(out['max_loss'],.06+1e-9)
    def test_net_losses_halt_and_missing_model_no_elite(self):
        self.assertEqual(daily_scan.regime_state({'n':40,'rule_avg_return':-.01,'hit_rate':.9}),'halt')
        self.assertEqual(daily_scan.get_elite_picks([{'tech_score':99}],False),([],[]))
    def test_compare_empty_and_costed_portfolio(self):
        self.assertEqual(compare([], {})['status'],'insufficient_data')
        r={'ticker':'ABC','tech_score':80,'combined':80,'pred':{'prob':.8,'passes':True},
           'days_to_earnings':60, 'atr_rel':.05}
        snap={'session':'2026-01-02','model_id':'test','rows':[r], 'execution_policies':{'ABC':execution.policy(.05)}}
        out=compare([snap], {'ABC':bars()})
        self.assertEqual(out['status'],'ok')
        self.assertLess(out['variants']['model']['net_return'],0)
        self.assertEqual(out['variants']['model']['n_trades'],1)

if __name__=='__main__': unittest.main()

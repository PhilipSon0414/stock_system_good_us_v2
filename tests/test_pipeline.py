import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import daily_scan
import surge_model
import tracker
import quiet_tracker
from market_clock import session_dates, session_offset
from data_fetcher import FetchLog
from execution import policy

class Pipeline(unittest.TestCase):
    def test_zero_candidate_report_and_duplicate_skip(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(daily_scan,'REPORTS_DIR',Path(d)), \
                 patch.object(daily_scan,'resolve_session',return_value='2026-10-26'), \
                 patch.object(daily_scan,'run_scan',return_value=([],False,FetchLog(),[])) as scan, \
                 patch.object(quiet_tracker,'save_quiet'), patch.object(quiet_tracker,'evaluate'), \
                 patch.object(tracker,'evaluate'), patch.object(tracker,'save_predictions'), \
                 patch.object(daily_scan,'send_report') as send:
                daily_scan.main()
                daily_scan.main()
                self.assertEqual(scan.call_count,1)
                self.assertEqual(send.call_count,1)
                report=json.loads((Path(d)/'2026-10-26_scan.json').read_text())
                self.assertEqual(report['results'],[])
                self.assertEqual(report['entry_basis'],'next_session_open')
    def test_calibration_training_smoke_and_abstention(self):
        dates=session_dates('2022-01-03','2025-12-31')
        rng=np.random.default_rng(42)
        data=pd.DataFrame({'date':np.repeat(dates,3),'signal':rng.normal(size=len(dates)*3)})
        data['label']=(data.signal>0).astype(int)
        with patch.object(surge_model,'TARGET_PRECISION',1.1):
            fit=surge_model._fit_calibrated(data,['signal'])
        self.assertEqual(fit['threshold'],1.01)
        self.assertFalse(fit['model'].early_stopping)
        self.assertTrue(np.isfinite(fit['iso'].predict([0,.5,1])).all())
    def test_trade_label_uses_next_open(self):
        dates=session_dates('2024-01-02','2025-03-31')[:280]
        df=pd.DataFrame({'Open':100.,'High':101.,'Low':99.,'Close':100.,'ATRRel':.05},index=dates)
        df.iloc[251:,df.columns.get_indexer(['Open','High','Low','Close'])]=[112,113,110,112]
        labels=surge_model.make_trade_labels(df)
        self.assertEqual(labels.iloc[250],0)
        self.assertTrue(labels.iloc[-20:].isna().all())
    def test_tracker_rebases_split_and_preserves_legacy(self):
        signal='2026-01-02'
        dates=session_dates(signal,session_offset(signal,20))
        # Stored signal 100; provider later adjusts every bar to 50 after split.
        df=pd.DataFrame({'Open':50.,'High':50.5,'Low':49.5,'Close':50.,'Volume':1000000},index=dates)
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); pred=root/'v3';pred.mkdir()
            legacy=root/'preds_2026-01-02.jsonl';legacy.write_text('legacy data')
            with patch.object(tracker,'PRED_DIR',pred), patch.object(tracker,'STATS_PATH',root/'stats.json'), \
                 patch.object(tracker,'get_ohlcv',return_value=df), patch.object(tracker,'completed_session',return_value='2026-02-10'):
                tracker.save_predictions([{'ticker':'ABC','close':100,'atr_rel':.05,'prob':.6,'elite':True}],signal)
                tracker.evaluate(False)
                r=json.loads(next(pred.glob('*.jsonl')).read_text())
                self.assertAlmostEqual(r['evals']['h20']['entry_price'],100.1)
                self.assertLess(r['evals']['h20']['rule_return'],0)
                before=r['evals']
                with patch.object(tracker,'COACH_TARGET',.5): tracker.evaluate(False)
                self.assertEqual(json.loads(next(pred.glob('*.jsonl')).read_text())['evals'],before)
                self.assertEqual(legacy.read_text(),'legacy data')
    def test_unknown_earnings_excluded(self):
        self.assertIn('실적 발표 일정 확인 필요',daily_scan.elite_exclusions({}))
    def test_quiet_policy_immutable(self):
        row={'ticker':'ABC','price':100,'vol_surge_3m':2.1,'price_3m':.02,'atr_rel':.05}
        with tempfile.TemporaryDirectory() as d, patch.object(quiet_tracker,'PRED_DIR',Path(d)):
            self.assertEqual(quiet_tracker.save_quiet([row],'2026-01-02'),1)
            file=next(Path(d).glob('*.jsonl'));before=file.read_text()
            self.assertEqual(quiet_tracker.save_quiet([row],'2026-01-02'),0)
            self.assertEqual(file.read_text(),before)
            self.assertEqual(json.loads(before)['execution_policy']['target'],.12)

if __name__=='__main__': unittest.main()

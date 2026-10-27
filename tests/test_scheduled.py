import tempfile
import unittest
import runpy
import sys
from pathlib import Path
from unittest.mock import patch
import scheduled_scan as runner

class Scheduled(unittest.TestCase):
    def test_existing_daily_command_uses_scheduled_entry(self):
        with patch.object(sys, 'argv', ['daily_scan.py','ALL']), patch.object(runner,'main') as scheduled:
            runpy.run_path(str(runner.HERE/'daily_scan.py'), run_name='__main__')
            scheduled.assert_called_once_with(['ALL'])

    def test_prepare_fresh_artifacts_no_training(self):
        with patch.object(runner,'model_ready',return_value=True), patch.object(runner,'rules_ready',return_value=True), patch.object(runner,'stale',return_value=False), patch.object(runner.subprocess,'run') as run:
            runner.prepare()
            run.assert_not_called()
    def test_prepare_missing_artifacts(self):
        with patch.object(runner,'model_ready',side_effect=[False,True]), patch.object(runner,'rules_ready',side_effect=[False,True]), patch.object(runner.subprocess,'run') as run:
            runner.prepare()
            self.assertEqual(run.call_count,2)
    def test_failure_does_not_scan(self):
        with patch.object(runner,'model_ready',return_value=False), patch.object(runner.subprocess,'run'):
            with self.assertRaises(RuntimeError): runner.prepare()
    def test_check_never_prepares(self):
        with patch.object(runner,'prepare') as prepare, patch.object(runner,'model_ready',return_value=True), patch.object(runner,'rules_ready',return_value=True):
            runner.main(['--check'])
            prepare.assert_not_called()
    def test_already_published_session_skips(self):
        with tempfile.TemporaryDirectory() as d, patch.object(runner,'HERE',Path(d)), patch.object(runner,'completed_session',return_value='2026-10-26'), patch.object(runner,'prepare') as prepare:
            (Path(d)/'reports').mkdir()
            (Path(d)/'reports'/'2026-10-26_scan.json').write_text('{}')
            runner.main([])
            prepare.assert_not_called()

if __name__=='__main__': unittest.main()

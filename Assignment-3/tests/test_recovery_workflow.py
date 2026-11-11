"""CPU-only checks for disk retention and local resume, without torch imports."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from common.recovery import prune_older_recoveries
from kaggle import runall


class RecoveryWorkflowTests(unittest.TestCase):
    def test_retention_preserves_midpoint_and_current(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / 'recovery_0312.pt'
            current = root / 'recovery_0625.pt'
            latest = root / 'latest.pt'
            midpoint = root / 'inference_0625.pt'
            for p in (old, current, midpoint):
                p.write_bytes(b'checkpoint')
            latest.hardlink_to(current)
            self.assertEqual(prune_older_recoveries(current, latest), [old.name])
            self.assertTrue(current.exists())
            self.assertTrue(midpoint.exists())
            self.assertEqual(latest.read_bytes(), b'checkpoint')

    def test_incomplete_publish_keeps_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, current, latest = [root / n for n in ('recovery_0312.pt', 'recovery_0625.pt', 'latest.pt')]
            old.write_bytes(b'old')
            current.write_bytes(b'new')
            latest.hardlink_to(old)
            self.assertEqual(prune_older_recoveries(current, latest), [])
            self.assertTrue(old.exists())

    def test_rolling_peak_and_time_budget(self):
        config = {'checkpoint_updates': [312, 625, 937, 1250]}
        gib = 2**30
        # Two recovery slots plus midpoint/final snapshots, not four recovery slots.
        self.assertEqual(runall.choose_stop_update(config, 'full', 0, 19*gib, 5*gib), 1250)
        self.assertIsNone(runall.choose_stop_update(config, 'full', 937, 4*gib, 5*gib))
        self.assertEqual(runall.choose_stop_update(config, 'full', 0, 100*gib, 5*gib, 10, 3200), 312)

    def test_verified_pending_local_resume_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = runall.Context(Path(__file__).resolve().parents[1], Path(directory))
            checkpoint = Path(directory) / 'recovery_0937.pt'
            checkpoint.write_bytes(b'valid')
            pending = {'method': 'full', 'step': 937, 'sha256': runall.sha256_file(checkpoint)}
            runall.atomic_json(ctx.state_path, {'backup_pending': pending})
            with patch.object(runall, '_checkpoint_progress', return_value=(937, checkpoint)):
                with self.assertRaises(runall.Blocked):
                    runall.resume_pending_locally(ctx, {}, 'l2sp')
                checkpoint.write_bytes(b'changed')
                with self.assertRaises(runall.Blocked):
                    runall.resume_pending_locally(ctx, {}, 'full')
                self.assertIn('backup_pending', ctx.state())
                checkpoint.write_bytes(b'valid')
                runall.resume_pending_locally(ctx, {}, 'full')
            self.assertNotIn('backup_pending', ctx.state())

    def test_measured_timing_ignores_partial_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.jsonl'
            path.write_text(json.dumps({'update': 10, 'training_seconds': 100}) + '\n' +
                            json.dumps({'update': 20, 'training_seconds': 130}) + '\n{')
            self.assertEqual(runall.measured_seconds_per_update(path), 3)


if __name__ == '__main__':
    unittest.main()

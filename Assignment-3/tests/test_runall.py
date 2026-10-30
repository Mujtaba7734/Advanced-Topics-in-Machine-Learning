"""Fast, GPU-free regression tests for the Kaggle stage controller."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat

from kaggle import runall
from kaggle.build_notebook import STAGES as NOTEBOOK_STAGES


ROOT = Path(__file__).resolve().parents[1]


class RunAllTests(unittest.TestCase):
    def test_isolated_interpreter_can_bootstrap_runner_and_verify_source(self):
        script = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
                  "from kaggle.runall import source_revision; print(source_revision(Path(sys.argv[1])))")
        result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT)],
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), runall.source_revision(ROOT))

    def test_environment_stage_runs_without_local_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            result = runall.run_stage("environment", ROOT, Path(directory))
            self.assertIn("source_revision", result)
            self.assertEqual(runall.read_json(Path(directory) / "outputs/task1/stage_state.json")
                             ["stages"]["environment"]["status"], "PASS")

    def test_notebook_has_independent_ordered_cells_and_matching_manifest(self):
        notebook = nbformat.read(ROOT / "kaggle/ATML_PA3_Task1_RunAll.ipynb", as_version=4)
        nbf_hash = hashlib.sha256((ROOT / "kaggle/source_manifest.json").read_bytes()).hexdigest()
        self.assertEqual(notebook.metadata["task1_manifest_sha256"], nbf_hash)
        self.assertEqual([stage for _, stage in NOTEBOOK_STAGES], list(runall.STAGES))
        cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        self.assertEqual(len(cells), 18)
        for cell, (_, stage) in zip(cells, NOTEBOOK_STAGES):
            compile(cell.source, f"notebook:{stage}", "exec")
            self.assertIn(f'"{stage}"', cell.source)
            self.assertIn(nbf_hash, cell.source)
            self.assertIn("subprocess.run", cell.source)
        self.assertEqual(runall.source_revision(ROOT), notebook.metadata["task1_source_revision"])

    def test_source_manifest_rejects_a_changed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kaggle").mkdir()
            (root / "source.py").write_text("original", encoding="utf-8")
            files = {"source.py": runall.sha256_file(root / "source.py")}
            revision = hashlib.sha256(json.dumps(files, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
            manifest = {"revision": revision, "files": files}
            runall.atomic_json(root / "kaggle/source_manifest.json", manifest)
            self.assertEqual(runall.source_revision(root), revision)
            (root / "source.py").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                runall.source_revision(root)

    def test_dependency_probe_repairs_missing_import_and_torchvision(self):
        failed = type("Result", (), {"returncode": 1, "stderr": "operator torchvision::nms does not exist",
                                      "stdout": ""})()
        good = type("Result", (), {"returncode": 0, "stderr": "", "stdout": json.dumps({
            "python": runall.sys.executable, "cuda": False, "gpus": 0})})()
        with patch.object(runall, "mismatched_requirements", side_effect=[["lm-eval==0.4.13"], []]), \
                patch.object(runall, "import_probe", side_effect=[failed, good]), \
                patch.object(runall, "_pip") as pip, \
                patch.object(runall.importlib.metadata, "version", return_value="0.20"):
            result = runall.ensure_dependencies(ROOT)
        self.assertEqual(result["gpus"], 0)
        pip.assert_any_call("install", "--quiet", "lm-eval==0.4.13")
        pip.assert_any_call("uninstall", "--yes", "--quiet", "torchvision")

    def test_dependency_probe_repairs_broken_lm_eval_with_existing_metadata(self):
        failed = type("Result", (), {"returncode": 1,
                                      "stderr": "ModuleNotFoundError: No module named 'lm_eval'",
                                      "stdout": ""})()
        good = type("Result", (), {"returncode": 0, "stderr": "", "stdout": json.dumps({
            "python": runall.sys.executable, "cuda": False, "gpus": 0})})()
        with patch.object(runall, "mismatched_requirements", side_effect=[[], []]), \
                patch.object(runall, "import_probe", side_effect=[failed, good]), \
                patch.object(runall, "_pip") as pip:
            runall.ensure_dependencies(ROOT)
        pip.assert_called_once_with("install", "--quiet", "--force-reinstall", "--no-deps",
                                    "lm-eval==0.4.13")

    def test_checkpoint_planner_respects_disk_and_time(self):
        config = {"checkpoint_updates": [312, 625, 937, 1250]}
        gib = 1024**3
        self.assertEqual(runall.choose_stop_update(config, "full", 0, 9*gib, 5*gib), 312)
        self.assertIsNone(runall.choose_stop_update(config, "full", 0, 4*gib, 5*gib))
        self.assertEqual(runall.choose_stop_update(config, "lora4", 312, 100*gib, 1*gib,
                                                   seconds_per_update=10, seconds_remaining=4000), 625)
        self.assertEqual(runall.choose_stop_update(config, "lora16", 937, 5*gib, 1*gib), 1250)
        self.assertIsNone(runall.choose_stop_update(config, "oplora16", 937, 5*gib, 1*gib))

    def test_checkpoint_identity_and_complete_run_detection(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs/task1/full"
            root.mkdir(parents=True)
            config = {"output_root": str(root.parent), "checkpoint_updates": [312, 625, 937, 1250],
                      "optimizer_updates": 1250}
            manifest = {"selected_sha256": "dataset-identity"}
            checkpoint = root / "recovery_1250.pt"
            torch.save({"step": 1250, "method": "full", "data_sha256": "dataset-identity",
                        "config": config, "position": 40000}, checkpoint)
            with patch("task1.dataset.load_selected", return_value=([], manifest)):
                self.assertEqual(runall._checkpoint_progress(config, "full"), (1250, checkpoint))
                self.assertFalse(runall._method_complete(config, "full", 1250))
                (root / "inference_0625.pt").write_bytes(b"mid")
                (root / "inference_1250.pt").write_bytes(b"final")
                (root / "train.jsonl").write_text('{"update": 1250}\n', encoding="utf-8")
                self.assertTrue(runall._method_complete(config, "full", 1250))
                torch.save({"step": 1250, "method": "full", "data_sha256": "wrong",
                            "config": config, "position": 40000}, checkpoint)
                with self.assertRaises(runall.Blocked):
                    runall._checkpoint_progress(config, "full")

    def test_latest_alias_tracks_verified_newer_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "recovery_0312.pt"
            new = root / "recovery_0625.pt"
            old.write_bytes(b"old")
            new.write_bytes(b"new")
            (root / "latest.pt").write_bytes(b"old")
            runall._synchronize_latest(new, root)
            self.assertEqual((root / "latest.pt").read_bytes(), b"new")
            self.assertEqual(old.read_bytes(), b"old")

    def test_dataset_stage_reuses_verified_prepared_data(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = runall.Context(ROOT, Path(directory))
            verified = {"rows": 40000, "data_sha256": "original", "probe_counts": {"gsm8k": 32,
                                                                                   "hellaswag": 32}}
            with patch.object(runall, "_data_status", return_value=verified), \
                    patch("task1.prepare_data.prepare") as prepare:
                self.assertEqual(runall.stage_dataset(ctx, {}), verified)
                prepare.assert_not_called()

    def test_final_status_does_not_claim_missing_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = runall.Context(ROOT, Path(directory))
            with patch.object(runall, "_data_status", side_effect=FileNotFoundError), \
                    patch("task1.dataset.load_selected", side_effect=FileNotFoundError), \
                    patch.object(runall, "_checkpoint_progress", return_value=(0, None)), \
                    patch.object(runall, "_condition_status", return_value=[]), \
                    patch.object(runall, "_qualitative_status", return_value=[]):
                result = runall.stage_final_verify(ctx, {"result_root": str(ctx.results),
                                                          "output_root": str(ctx.output),
                                                          "optimizer_updates": 1250})
            self.assertFalse(result["computational_complete"])
            report = runall.read_json(ctx.results / "summary/final_verification.json")
            self.assertEqual(report["status"]["training_full"], "INCOMPLETE")
            self.assertEqual(report["status"]["student_written_report"], "NOT RUN")

    def test_restore_verifies_pending_checkpoint_and_preserves_work(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "work"
            mount = Path(directory) / "mount"
            recovery = mount / "outputs/task1/full/recovery_0312.pt"
            recovery.parent.mkdir(parents=True)
            recovery.write_bytes(b"synthetic recovery")
            prior = mount / "results/task1/evaluation/baseline/gsm8k_cot.json"
            prior.parent.mkdir(parents=True)
            prior.write_text("mounted", encoding="utf-8")
            ctx = runall.Context(ROOT, work)
            state = {"source_revision": ctx.revision, "stages": {
                "dataset": {"status": "PASS", "source_revision": ctx.revision}},
                "backup_pending": {"method": "full", "step": 312,
                                   "sha256": runall.sha256_file(recovery)}}
            runall.atomic_json(mount / "outputs/task1/stage_state.json", state)
            ctx.record("environment", "PASS", source_revision=ctx.revision)
            existing = work / "results/task1/evaluation/baseline/gsm8k_cot.json"
            existing.parent.mkdir(parents=True)
            existing.write_text("working", encoding="utf-8")
            with patch.object(ctx, "output_mounts", return_value=[mount]):
                result = runall.stage_restore(ctx)
            self.assertEqual(result["copied"], 1)
            self.assertEqual(existing.read_text(encoding="utf-8"), "working")
            self.assertEqual((work / "outputs/task1/full/recovery_0312.pt").read_bytes(),
                             b"synthetic recovery")
            self.assertNotIn("backup_pending", ctx.state())
            self.assertEqual(ctx.state()["stages"]["dataset"]["status"], "PASS")
            self.assertEqual(ctx.state()["stages"]["environment"]["status"], "PASS")

    def test_export_excludes_large_checkpoints_and_reuses_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            ctx = runall.Context(ROOT, work)
            checkpoint = ctx.output / "full/recovery_0312.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic checkpoint")
            result = ctx.results / "summary/benchmarks.csv"
            result.parent.mkdir(parents=True)
            result.write_text("score\n", encoding="utf-8")
            with patch.object(runall.shutil, "disk_usage", return_value=type("Disk", (), {"free": 5*1024**3})()):
                runall.stage_export(ctx, {})
                first_hash = runall.sha256_file(work / "task1_results_bundle.zip")
                runall.stage_export(ctx, {})
            self.assertEqual(first_hash, runall.sha256_file(work / "task1_results_bundle.zip"))
            import zipfile
            with zipfile.ZipFile(work / "task1_results_bundle.zip") as bundle:
                self.assertNotIn("outputs/task1/full/recovery_0312.pt", bundle.namelist())
                self.assertIn("results/task1/summary/backup_manifest.json", bundle.namelist())
            manifest = runall.read_json(ctx.results / "summary/backup_manifest.json")
            self.assertFalse(manifest["durable_backup_confirmed"])
            self.assertEqual(manifest["files"][0]["kind"], "recovery_checkpoint")

    def test_complete_stage_orchestration_with_lightweight_mocks(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            simple = lambda *args, **kwargs: {}
            with patch.object(runall, "ensure_dependencies", side_effect=simple), \
                    patch.object(runall.Context, "config", return_value={}), \
                    patch.object(runall, "stage_environment", side_effect=simple), \
                    patch.object(runall, "stage_restore", side_effect=simple), \
                    patch.object(runall, "stage_dataset", side_effect=simple), \
                    patch.object(runall, "stage_preflight", side_effect=simple), \
                    patch.object(runall, "stage_baseline", side_effect=simple), \
                    patch.object(runall, "stage_train", side_effect=simple) as train, \
                    patch.object(runall, "stage_evaluate_models", side_effect=simple), \
                    patch.object(runall, "stage_weight", side_effect=simple), \
                    patch.object(runall, "stage_activations", side_effect=simple), \
                    patch.object(runall, "stage_qualitative", side_effect=simple), \
                    patch.object(runall, "stage_aggregate", side_effect=simple), \
                    patch.object(runall, "stage_final_verify", side_effect=simple), \
                    patch.object(runall, "stage_export", side_effect=simple), \
                    patch("task1.dataset.load_selected", return_value=([], {})), \
                    patch("task1.preflight.require_passed_preflight", side_effect=simple):
                for stage in runall.STAGES:
                    runall.run_stage(stage, ROOT, work)
            state = runall.read_json(work / "outputs/task1/stage_state.json")
            self.assertEqual(set(state["stages"]), set(runall.STAGES))
            self.assertTrue(all(record["status"] == "PASS" for record in state["stages"].values()))
            self.assertEqual([call.args[2] for call in train.call_args_list], list(runall.METHODS))


if __name__ == "__main__":
    unittest.main()

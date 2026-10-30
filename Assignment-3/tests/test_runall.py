"""Fast, GPU-free regression tests for the Kaggle stage controller."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat

from kaggle import runall
from kaggle.build_notebook import (STAGES as NOTEBOOK_STAGES, next_clone_path, stage_cell,
                                   repository_candidates, source_matches)


ROOT = Path(__file__).resolve().parents[1]


class RunAllTests(unittest.TestCase):
    def test_generated_cell_bootstraps_fresh_and_reuses_content_addressed_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            working, input_root = root / "working", root / "input"
            working.mkdir()
            input_root.mkdir()
            source = root / "fixture/Assignment-3"
            (source / "kaggle").mkdir(parents=True)
            (source / "source.py").write_text("audited", encoding="utf-8")
            files = {"source.py": runall.sha256_file(source / "source.py")}
            revision = hashlib.sha256(json.dumps(files, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
            runall.atomic_json(source / "kaggle/source_manifest.json",
                               {"revision": revision, "files": files})
            expected = runall.sha256_file(source / "kaggle/source_manifest.json")
            stale = working / "atml-repo-audited/Assignment-3"
            stale.mkdir(parents=True)
            (stale / "sentinel.txt").write_text("preserve", encoding="utf-8")
            code = stage_cell("environment", expected)
            code = code.replace('Path("/kaggle/working")', f"Path({json.dumps(str(working))})")
            code = code.replace('Path("/kaggle/input")', f"Path({json.dumps(str(input_root))})")
            calls = []

            def fake_run(args, **kwargs):
                calls.append((args, kwargs))
                if args[:2] == ["git", "clone"]:
                    destination = Path(args[-1]) / "Assignment-3"
                    shutil.copytree(source, destination)
                return types.SimpleNamespace(returncode=0)

            with patch("subprocess.run", side_effect=fake_run):
                exec(compile(code, "notebook:environment", "exec"), {})
                exec(compile(code, "notebook:environment", "exec"), {})
            clones = [args for args, _ in calls if args[:2] == ["git", "clone"]]
            runner = [(args, kwargs) for args, kwargs in calls
                      if args[:3] == [sys.executable, "-m", "kaggle.runall"]]
            self.assertEqual(len(clones), 1)
            self.assertEqual(len(runner), 2)
            self.assertIn(expected[:16], clones[0][-1])
            self.assertEqual(runner[0][1]["cwd"], runner[1][1]["cwd"])
            self.assertEqual((stale / "sentinel.txt").read_text(encoding="utf-8"), "preserve")

    def test_stale_checkout_is_ignored_and_content_addressed_clone_is_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            working, input_root = root / "working", root / "input"
            stale = working / "atml-repo-audited/Assignment-3"
            stale.mkdir(parents=True)
            (stale / "kaggle").mkdir()
            (stale / "kaggle/source_manifest.json").write_text("{}", encoding="utf-8")
            mounted = input_root / "source/Assignment-3"
            (mounted / "kaggle").mkdir(parents=True)
            (mounted / "source.py").write_text("current", encoding="utf-8")
            files = {"source.py": runall.sha256_file(mounted / "source.py")}
            revision = hashlib.sha256(json.dumps(files, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
            runall.atomic_json(mounted / "kaggle/source_manifest.json",
                               {"revision": revision, "files": files})
            expected = runall.sha256_file(mounted / "kaggle/source_manifest.json")
            matches = [path for path in repository_candidates(working, input_root)
                       if source_matches(path, expected)]
            self.assertEqual(matches, [mounted])
            clone = next_clone_path(working, expected)
            self.assertEqual(clone.name, "atml-repo-" + expected[:16])
            clone.mkdir()
            self.assertEqual(next_clone_path(working, expected).name, clone.name + "-repair-1")
            self.assertTrue((stale / "kaggle/source_manifest.json").exists())

    def test_local_module_wins_over_installed_regular_kaggle_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_installed = root / "site-packages/kaggle"
            fake_installed.mkdir(parents=True)
            (fake_installed / "__init__.py").write_text("FAKE_STANDARD_KAGGLE = True\n", encoding="utf-8")
            environment = {**os.environ, "PYTHONPATH": str(fake_installed.parent)}
            work = root / "work"
            result = subprocess.run(
                [sys.executable, "-m", "kaggle.runall", "environment",
                 "--repo", str(ROOT), "--work", str(work)],
                cwd=ROOT, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("[PASS] Environment", result.stdout)
            self.assertTrue((work / "outputs/task1/stage_state.json").is_file())

    def test_bounded_smoke_uses_real_evaluation_entry_for_all_four_tasks(self):
        import task1.evaluation as evaluation
        from task1.plots import PRIMARY
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root / "official/evaluation/baseline/gsm8k_cot.json"
            official.parent.mkdir(parents=True)
            official.write_text("keep official result", encoding="utf-8")
            config = {"model": "Qwen/Qwen3-0.6B", "model_revision": None,
                      "output_root": str(root / "outputs"), "result_root": str(root / "official"),
                      "cache_dir": str(root / "cache"), "seed": 6304,
                      "gsm8k_max_new_tokens": 1024}
            calls = []

            def fake_simple_evaluate(**kwargs):
                task = kwargs["tasks"][0]
                calls.append((task, kwargs["limit"], kwargs.get("gen_kwargs")))
                sample = {"filtered_resps": [[1.0], [0.5]], "doc": {"choices": ["A", "B"]}}
                if task == "gsm8k_cot":
                    sample = {"filter": "flexible-extract", "filtered_resps": ["42"],
                              "resps": [["42"]], "doc": {}}
                elif task == "arc_easy":
                    sample["doc"] = {"choices": {"text": ["A", "B"]}}
                return {"samples": {task: [dict(sample), dict(sample)]},
                        "results": {task: {PRIMARY[task]: 0.5}}, "config": {}}

            fake_lm_eval = types.ModuleType("lm_eval")
            fake_lm_eval.__path__ = []
            fake_lm_eval.simple_evaluate = fake_simple_evaluate
            fake_models = types.ModuleType("lm_eval.models")
            fake_models.__path__ = []
            fake_hf = types.ModuleType("lm_eval.models.huggingface")
            fake_hf.HFLM = lambda **kwargs: object()
            fake_tokenizer = types.SimpleNamespace(encode=lambda *args, **kwargs: [1])
            modules = {"lm_eval": fake_lm_eval, "lm_eval.models": fake_models,
                       "lm_eval.models.huggingface": fake_hf}
            with patch.dict(sys.modules, modules), \
                    patch.object(evaluation.AutoTokenizer, "from_pretrained", return_value=fake_tokenizer), \
                    patch.object(evaluation, "load_condition", return_value=object()), \
                    patch.object(evaluation, "evaluate", wraps=evaluation.evaluate) as actual_entry:
                runall.bounded_evaluation_smoke(config, "baseline", root)
                actual_entry.assert_called_once()
            self.assertEqual([task for task, _, _ in calls], list(runall.TASKS))
            self.assertTrue(all(limit == 2 for _, limit, _ in calls))
            self.assertEqual(calls[0][2], {"max_gen_toks": 64, "do_sample": False})
            self.assertTrue(all(options is None for _, _, options in calls[1:]))
            self.assertEqual(official.read_text(encoding="utf-8"), "keep official result")

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

    def test_every_manifest_hash_matches_source_bytes(self):
        manifest = runall.read_json(ROOT / "kaggle/source_manifest.json")
        self.assertIn("task1/preflight.py", manifest["files"])
        self.assertIn("kaggle/runall.py", manifest["files"])
        self.assertIn("kaggle/__init__.py", manifest["files"])
        for relative, expected in manifest["files"].items():
            self.assertEqual(runall.sha256_file(ROOT / relative), expected, relative)
        calculated = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(manifest["revision"], calculated)

    def test_old_successful_preflight_is_reused_when_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = runall.Context(ROOT, Path(directory))
            with patch("task1.dataset.load_selected", return_value=([], {"selected_sha256": "data"})), \
                    patch("task1.preflight.require_passed_preflight") as require, \
                    patch("task1.preflight.run_preflight") as preflight:
                result = runall.stage_preflight(ctx, {})
            self.assertEqual(result["status"], "verified_existing")
            require.assert_called_once()
            preflight.assert_not_called()

    def test_complete_baseline_skips_smoke_and_recalculation(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = runall.Context(ROOT, Path(directory))
            with patch.object(runall, "_condition_status", return_value=list(runall.TASKS)), \
                    patch.object(runall, "bounded_evaluation_smoke") as smoke, \
                    patch("task1.evaluation.evaluate") as evaluate:
                result = runall.stage_baseline(ctx, {})
            self.assertEqual(result["status"], "verified_existing")
            smoke.assert_not_called()
            evaluate.assert_not_called()

    def test_complete_evaluation_requires_matching_identity_metric_and_predictions(self):
        from task1.plots import PRIMARY
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"result_root": str(root)}
            tasks = {name: 2 for name in runall.TASKS}
            location = root / "evaluation/baseline"
            location.mkdir(parents=True)
            for task in tasks:
                runall.atomic_json(location / f"{task}.json", {
                    "condition": "baseline", "task": task, "condition_fingerprint": "model",
                    "sample_count": 2, "results": {PRIMARY[task]: 0.5},
                    "samples": [{"prediction": 0, "prediction_raw": 0,
                                 "truncation_flag": False} for _ in range(2)]})
            with patch.object(runall, "TASKS", tasks), \
                    patch("task1.evaluation.condition_fingerprint", return_value="model"):
                self.assertEqual(runall._condition_status(config, "baseline"), list(tasks))
                corrupt = runall.read_json(location / "boolq.json")
                corrupt["samples"][0].pop("prediction")
                runall.atomic_json(location / "boolq.json", corrupt)
                with self.assertRaises(runall.Blocked):
                    runall._condition_status(config, "baseline")
                corrupt["samples"][0]["prediction"] = 0
                corrupt["sample_count"] = 1
                runall.atomic_json(location / "boolq.json", corrupt)
                self.assertEqual(len(runall._condition_status(config, "baseline")), 3)
                corrupt["condition_fingerprint"] = "wrong"
                runall.atomic_json(location / "boolq.json", corrupt)
                with self.assertRaises(runall.Blocked):
                    runall._condition_status(config, "baseline")

    def test_six_model_evaluations_select_every_required_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = runall.Context(ROOT, Path(directory))
            counts = {}

            def status(config, condition):
                counts[condition] = counts.get(condition, 0) + 1
                return [] if counts[condition] == 1 else list(runall.TASKS)

            with patch.object(runall, "_condition_status", side_effect=status), \
                    patch("task1.evaluation.evaluate") as evaluate:
                result = runall.stage_evaluate_models(ctx, {})
            self.assertEqual(list(result), list(runall.CONDITIONS[1:]))
            self.assertEqual([call.args[1] for call in evaluate.call_args_list],
                             list(runall.CONDITIONS[1:]))

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
            manifest_sha = runall.sha256_file(root / "kaggle/source_manifest.json")
            self.assertTrue(source_matches(root, manifest_sha))
            (root / "source.py").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                runall.source_revision(root)
            self.assertFalse(source_matches(root, manifest_sha))

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
                torch.save({"layer.weight": torch.ones(1)}, root / "inference_0625.pt")
                torch.save({"layer.weight": torch.ones(1)}, root / "inference_1250.pt")
                (root / "train.jsonl").write_text('{"update": 1250}\n', encoding="utf-8")
                self.assertTrue(runall._method_complete(config, "full", 1250))
                (root / "inference_1250.pt").write_bytes(b"corrupt")
                self.assertFalse(runall._method_complete(config, "full", 1250))
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

    def test_partial_training_records_backup_gate_and_blocks_next_method(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            ctx = runall.Context(ROOT, work)
            recovery = ctx.output / "full/recovery_0312.pt"
            recovery.parent.mkdir(parents=True)
            recovery.write_bytes(b"checkpoint bytes")
            config = {"output_root": str(ctx.output), "checkpoint_updates": [312, 625, 937, 1250],
                      "optimizer_updates": 1250}
            with patch.object(runall, "_checkpoint_progress", side_effect=[(0, None), (312, recovery)]), \
                    patch.object(runall, "_method_complete", return_value=False), \
                    patch.object(runall, "_checkpoint_size_from_preflight", return_value=(100, {})), \
                    patch.object(runall, "choose_stop_update", return_value=312), \
                    patch("task1.training.run_training", return_value={"complete": False}):
                result = runall.stage_train(ctx, config, "full")
            self.assertTrue(result["backup_pending"])
            self.assertEqual(ctx.state()["backup_pending"]["sha256"], runall.sha256_file(recovery))
            with patch.object(runall, "ensure_dependencies", return_value={}), \
                    patch.object(runall.Context, "config", return_value=config):
                blocked = runall.run_stage("train_l2sp", ROOT, work)
            self.assertIn("Save Version", blocked["blocked"])
            self.assertEqual(ctx.state()["stages"]["train_l2sp"]["status"], "BLOCKED")

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
                self.assertIn("source/kaggle/runall.py", bundle.namelist())
                self.assertIn("source/kaggle/__init__.py", bundle.namelist())
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

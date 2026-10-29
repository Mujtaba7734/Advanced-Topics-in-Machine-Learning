import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn

from common.checkpoint import atomic_save, atomic_latest, load_checked
from common.seed import set_seed, rng_state, restore_rng
from task1.prepare_data import normalize_question, select_examples, encode_example
from task1.training import token_loss_sum, response_loss_sum, weight_decay_groups, trim_jsonl_to_checkpoint
from task1.model import AdapterLinear, leading_bases, trainable_parameters, projected_adapter_state, use_projected_factors
from task1.l2sp import CPUReference, unique_named_parameters
from task1.diagnostics import metrics, activation_ratio
from task1.evaluation import answer_from_sample, select_task_samples, multiple_choice_prediction
from task1.qualitative import load_prompts, prompt_sha256, combine_outputs, annotation_template
from task1.evaluation import CONDITIONS
from task1.plots import aggregate
from task1.dataset import load_selected


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not enable_thinking and tokenize
        user = [1, 2] + [ord(c) for c in messages[0]["content"]] + [3]
        if add_generation_prompt:
            return user + [4]
        return user + [4] + [ord(c) for c in messages[1]["content"]] + [5]


class DataTests(unittest.TestCase):
    def test_selection_masking_and_order(self):
        tok = FakeTokenizer()
        rows = [{"query": "  ", "response": "a"},
                {"query": "First!", "original_question": "GSM  8K?", "response": "a"},
                {"query": "First!", "response": "answer"},
                {"query": "too long", "response": "x" * 100},
                {"query": "Last?", "response": "b"}]
        selected, rejected = select_examples(rows, ["gsm 8k"], tok, count=2, max_length=30)
        self.assertEqual([r["source_index"] for r in selected], [2, 4])
        self.assertEqual(rejected, {"empty": 1, "contaminated": 1, "too_long": 1})
        self.assertEqual(normalize_question("GSM  8K?"), normalize_question("gsm 8k"))
        for row in selected:
            self.assertEqual(len(row["input_ids"]), len(row["labels"]))
            self.assertLessEqual(len(row["input_ids"]), 30)
            self.assertEqual(sum(x != -100 for x in row["labels"]), row["response_tokens"])
            self.assertEqual(row["input_ids"][-1], row["labels"][-1])

    def test_empty_and_too_long_fail(self):
        with self.assertRaises(RuntimeError):
            select_examples([{"query": "q", "response": ""}], [], FakeTokenizer(), count=1)

    def test_saved_data_integrity_and_order(self):
        import hashlib
        with tempfile.TemporaryDirectory() as root:
            data = Path(root) / "data"
            data.mkdir()
            rows = [{"source_index": 2, "input_ids": [1], "labels": [1]},
                    {"source_index": 5, "input_ids": [2], "labels": [2]}]
            payload = json.dumps(rows).encode()
            (data / "selected.json").write_bytes(payload)
            manifest = {"selected_sha256": hashlib.sha256(payload).hexdigest(), "indices": [2, 5]}
            (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            config = {"output_root": root, "train_examples": 2}
            loaded, _ = load_selected(config)
            self.assertEqual([r["source_index"] for r in loaded], [2, 5])
            (data / "selected.json").write_bytes(payload + b" ")
            with self.assertRaises(ValueError):
                load_selected(config)


class ObjectiveTests(unittest.TestCase):
    def test_token_normalization(self):
        logits = torch.randn(2, 5, 7, requires_grad=True)
        labels = torch.tensor([[-100, -100, 3, 1, 2], [-100, -100, -100, -100, 4]])
        total = token_loss_sum(logits, labels)
        pieces = sum(token_loss_sum(logits[i:i+1], labels[i:i+1]) for i in range(2))
        self.assertTrue(torch.allclose(total / 4, pieces / 4))
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_selected_hidden_loss_matches_full_logits(self):
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(9, 4)
                self.lm_head = nn.Linear(4, 9, bias=False)
                self.model = self
            def forward(self, input_ids, attention_mask, use_cache):
                return type("Output", (), {"last_hidden_state": self.embedding(input_ids)})()
        model = TinyLM()
        ids = torch.tensor([[1, 2, 3, 4], [3, 2, 1, 5]])
        labels = torch.tensor([[-100, -100, 3, 4], [-100, -100, -100, 5]])
        full = token_loss_sum(model.lm_head(model.embedding(ids)), labels)
        sparse = response_loss_sum(model, ids, torch.ones_like(ids), labels, chunk_size=1)
        self.assertTrue(torch.allclose(full, sparse, atol=1e-6))

    def test_l2sp_and_tied_parameters(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Parameter(torch.tensor([[1., 2.]]))
                self.b = self.a
                self.bias = nn.Parameter(torch.tensor([3.]))
        model = Tiny()
        self.assertEqual(len(list(unique_named_parameters(model))), 2)
        ref = CPUReference(model)
        with torch.no_grad():
            model.a.add_(1)
            model.bias.add_(2)
        omega = ref.omega_and_gradient(model, coefficient=0.1)
        self.assertAlmostEqual(omega, 3.0)
        self.assertTrue(torch.allclose(model.a.grad, torch.full_like(model.a, 0.1)))
        self.assertTrue(torch.allclose(model.bias.grad, torch.tensor([0.2])))
        groups = weight_decay_groups(model, 0.01)
        self.assertEqual(len(groups[0]["params"]), 1)
        self.assertEqual(len(groups[1]["params"]), 1)


class AdapterTests(unittest.TestCase):
    def test_counts_initialization_gradients_and_freeze(self):
        set_seed(6304)
        base4 = nn.Linear(7, 5)
        initial = base4.weight.detach().clone()
        a4 = AdapterLinear(base4, 4, 8, 0.05)
        a16 = AdapterLinear(nn.Linear(7, 5), 16, 32, 0.05)
        self.assertEqual(trainable_parameters(a16), 4 * trainable_parameters(a4))
        self.assertTrue(torch.equal(a4.correction(), torch.zeros_like(initial)))
        out = a4(torch.randn(3, 7)).sum()
        out.backward()
        self.assertEqual(tuple(a4(torch.randn(3, 7)).shape), (3, 5))
        self.assertIsNone(base4.weight.grad)
        self.assertTrue(torch.equal(base4.weight, initial))
        self.assertIsNotNone(a4.B.grad)

    def test_projection_orthogonality_merge(self):
        torch.manual_seed(9)
        base = nn.Linear(6, 5, bias=False)
        uk, vk = leading_bases(base.weight, 2)
        adapter = AdapterLinear(base, 2, 4, 0.0, uk, vk)
        with torch.no_grad():
            adapter.B.normal_()
        b, a = adapter.effective_factors()
        delta = adapter.correction()
        self.assertEqual(tuple(delta.shape), (5, 6))
        self.assertLess((uk.T @ delta).abs().max().item(), 1e-5)
        self.assertLess((delta @ vk).abs().max().item(), 1e-5)
        x = torch.randn(4, 6)
        adapter.eval()
        self.assertTrue(torch.allclose(adapter(x), nn.functional.linear(x, base.weight + delta), atol=1e-5))
        adapter(x).sum().backward()
        self.assertIsNotNone(adapter.A.grad)
        self.assertIsNotNone(adapter.B.grad)
        self.assertIsNone(base.weight.grad)

    def test_projected_factor_export_equivalence(self):
        class Wrapped(nn.Module):
            def __init__(self):
                super().__init__()
                weight = nn.Linear(6, 5, bias=False)
                uk, vk = leading_bases(weight.weight, 2)
                self.projection = AdapterLinear(weight, 2, 4, 0.0, uk, vk)
        model = Wrapped()
        with torch.no_grad():
            model.projection.B.normal_()
        model.eval()
        x = torch.randn(3, 6)
        before = model.projection(x)
        exported = projected_adapter_state(model)
        self.assertEqual(set(exported), {"projection.A", "projection.B"})
        use_projected_factors(model, exported)
        self.assertTrue(torch.allclose(before, model.projection(x), atol=1e-5))


class RecoveryAndMetricTests(unittest.TestCase):
    def test_resume_and_checkpoint_integrity(self):
        set_seed(6304)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "recovery.pt"
            original = {"method": "tiny", "step": 312, "position": 9984, "data_sha256": "abc",
                        "rng": rng_state(), "weight": torch.tensor([2.])}
            expected_next = torch.rand(1)
            atomic_save(original, path)
            latest = Path(root) / "latest.pt"
            atomic_latest(path, latest)
            self.assertEqual(load_checked(latest, {"step": 312})["position"], 9984)
            state = load_checked(path, {"method": "tiny", "data_sha256": "abc"})
            self.assertEqual(state["position"], state["step"] * 32)
            restore_rng(state["rng"])
            self.assertTrue(torch.equal(torch.rand(1), expected_next))
            with self.assertRaises(ValueError):
                load_checked(path, {"data_sha256": "changed"})
            self.assertFalse(path.with_name("recovery.pt.tmp").exists())
            log = Path(root) / "train.jsonl"
            log.write_text('{"update":312}\n{"update":313}\n{"update":', encoding="utf-8")
            trim_jsonl_to_checkpoint(log, "update", 312)
            self.assertEqual(log.read_text(encoding="utf-8"), '{"update":312}\n')

    def test_weight_and_activation_metrics(self):
        w = torch.eye(3)
        delta = torch.diag(torch.tensor([1., 0., 0.]))
        uk = torch.eye(3)[:, :1]
        m = metrics(w, delta, uk)
        self.assertAlmostEqual(m["rho"], 1.0)
        self.assertAlmostEqual(m["eta"], 1 / np.sqrt(3))
        self.assertEqual(metrics(w, torch.zeros_like(w), uk)["rho"], 0)
        ratio = activation_ratio(w, delta, torch.tensor([[1., 0., 0.]]))
        self.assertAlmostEqual(ratio.item(), 1.0)

    def test_harness_answer_extraction(self):
        self.assertEqual(answer_from_sample({"filtered_resps": ["42"]}), "42")
        self.assertIsNone(answer_from_sample({"filtered_resps": []}))
        samples = [{"filter": "strict-match"}, {"filter": "flexible-extract"}]
        self.assertEqual(len(select_task_samples("gsm8k_cot", samples)), 1)
        self.assertEqual(multiple_choice_prediction({"filtered_resps": [(-2.0, False), (-1.0, True)]}), 1)

    def test_qualitative_assignment_style_schema(self):
        import yaml
        data = []
        for suffix in ("", "_json", "_template"):
            for i in range(8):
                data.append({"id": f"question_{i}{suffix}", "paraphrases": [f"Prompt {i}"],
                             "system": "Return JSON" if suffix == "_json" else ""})
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "first_plot_questions.yaml"
            path.write_text(yaml.safe_dump(data), encoding="utf-8")
            rows, ids = load_prompts(path)
        self.assertEqual(len(rows), 24)
        self.assertEqual(len(ids), 8)
        self.assertEqual({r["format"] for r in rows}, {"plain", "json", "template"})
        self.assertEqual(sum(bool(r["system"]) for r in rows), 8)

    def test_attached_qualitative_file_and_annotation_cardinality(self):
        official = Path(__file__).resolve().parents[1] / "first_plot_questions.yaml"
        rows, ids = load_prompts(official)
        self.assertEqual((len(rows), len(ids)), (24, 8))
        self.assertEqual(sum(bool(r["system"]) for r in rows), 8)
        fingerprint = prompt_sha256(official)
        with tempfile.TemporaryDirectory() as root:
            qualitative = Path(root) / "qualitative"
            qualitative.mkdir()
            for condition in CONDITIONS:
                payload = [{**row, "condition": condition, "prompt_sha256": fingerprint,
                            "response": "synthetic test response", "generated_tokens": 1,
                            "truncated": False} for row in rows]
                (qualitative / f"{condition}.json").write_text(json.dumps(payload), encoding="utf-8")
            config = {"result_root": root, "qualitative_prompts": str(official)}
            combined = combine_outputs(config)
            annotation = annotation_template(config)
            import pandas as pd
            self.assertEqual(len(pd.read_csv(combined)), 168)
            self.assertEqual(len(pd.read_csv(annotation)), 42)
            changed = json.loads((qualitative / "baseline.json").read_text(encoding="utf-8"))
            changed[0]["prompt"] = "changed"
            (qualitative / "baseline.json").write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                combine_outputs(config)

    def test_incomplete_aggregation_uses_recorded_values(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "outputs"
            results = Path(root) / "results"
            method = output / "full"
            method.mkdir(parents=True)
            (method / "train.jsonl").write_text(json.dumps({
                "update": 625, "sft_loss": 1.25, "training_seconds": 4.0,
                "peak_gpu_bytes": 1024, "reference_bytes": 0,
                "reference_transfer_seconds": 0.0}) + "\n", encoding="utf-8")
            (method / "model_metadata.json").write_text(
                json.dumps({"trainable_parameters": 12}), encoding="utf-8")
            for condition, score in (("baseline", 0.5), ("full_mid", 0.6)):
                evaluation = results / "evaluation" / condition
                evaluation.mkdir(parents=True)
                (evaluation / "gsm8k_cot.json").write_text(json.dumps({
                    "sample_count": 1319, "results": {"exact_match,flexible-extract": score}}),
                    encoding="utf-8")
            summary = aggregate({"output_root": str(output), "result_root": str(results)})
            resources = __import__("pandas").read_csv(summary / "resources.csv")
            benchmark = __import__("pandas").read_csv(summary / "benchmarks.csv")
            self.assertEqual(resources.iloc[0]["status"], "incomplete")
            self.assertEqual(resources.iloc[0]["trainable_parameters"], 12)
            self.assertAlmostEqual(benchmark.loc[benchmark.condition == "full_mid", "change_pp"].iloc[0], 10)
            self.assertTrue((summary / "sft_loss.png").exists())


if __name__ == "__main__":
    unittest.main()

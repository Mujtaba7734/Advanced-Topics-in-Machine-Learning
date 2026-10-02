from __future__ import annotations

import argparse

import pandas as pd
from tqdm.auto import tqdm

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer


POLICY_ORDER = ("sft", "dpo", "ppo", "grpo")


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    df = pd.read_csv(repo_path(cfg["paths"]["xstest"]))
    required = {"xstest_id", "prompt", "benchmark_class", "type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"XSTest CSV missing required columns: {sorted(missing)}"
        )
    ids = df["xstest_id"].astype(int).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("XSTest contains duplicate xstest_id values")
    classes = set(df["benchmark_class"].astype(str).str.upper())
    if not classes.issubset({"SAFE", "UNSAFE"}):
        raise ValueError(f"Unexpected XSTest benchmark classes: {sorted(classes)}")
    return df


def _expected_record(row, policy_name: str):
    return {
        "xstest_id": int(row["xstest_id"]),
        "policy": policy_name,
        "prompt": str(row["prompt"]),
        "benchmark_class": str(row["benchmark_class"]).upper(),
        "type": str(row["type"]),
    }


def _validate_existing_prefix(existing, df, policy_name: str):
    if len(existing) > len(df):
        raise ValueError(
            f"Saved {policy_name} generation has {len(existing)} rows, "
            f"but XSTest has only {len(df)}"
        )
    for i, saved in enumerate(existing):
        expected = _expected_record(df.iloc[i], policy_name)
        for key, value in expected.items():
            if str(saved.get(key)) != str(value):
                raise ValueError(
                    f"Saved {policy_name} generation is not the fixed XSTest "
                    f"prefix at row {i}: {key}={saved.get(key)!r}, "
                    f"expected {value!r}"
                )
        if "response" not in saved or "response_tokens" not in saved:
            raise ValueError(
                f"Saved {policy_name} row {i} is missing response fields"
            )


def generate_for_policy(
    cfg,
    policy_name: str,
    batch_size: int = 4,
    resume: bool = False,
):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    if batch_size <= 0:
        raise ValueError("safety batch size must be positive")

    df = load_xstest(cfg)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / f"generated_{policy_name}.jsonl"

    existing = read_jsonl(dst) if resume and dst.exists() else []
    _validate_existing_prefix(existing, df, policy_name)

    if len(existing) == len(df):
        print(
            f"[resume] Reusing complete deterministic responses: "
            f"{policy_name} ({len(existing)}/{len(df)})",
            flush=True,
        )
        return existing

    # Keep only a complete prefix. Generation is deterministic, so restarting
    # from any row is scientifically equivalent; rewriting once here avoids
    # carrying malformed trailing data from an interrupted write.
    write_jsonl(dst, existing)

    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(
        cfg,
        adapter_path=adapter,
        trainable=False,
    )

    try:
        records = list(existing)
        starts = list(range(len(existing), len(df), batch_size))
        progress = tqdm(
            starts,
            total=len(starts),
            desc=f"Task 4 generation: {policy_name}",
            unit="batch",
            dynamic_ncols=True,
        )

        for start in progress:
            chunk = df.iloc[start : start + batch_size]
            prompts = [
                [{"role": "user", "content": str(x)}]
                for x in chunk["prompt"].tolist()
            ]
            gen = batch_generate(
                model,
                tokenizer,
                prompts,
                max_prompt_length=int(cfg.get("safety_max_prompt_length", 256)),
                max_new_tokens=int(cfg["safety_max_new_tokens"]),
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )

            batch_records = []
            for (_, row), response, n_tok in zip(
                chunk.iterrows(),
                gen["responses"],
                gen["response_lengths"],
            ):
                record = {
                    **_expected_record(row, policy_name),
                    "response": response,
                    "response_tokens": int(n_tok),
                }
                batch_records.append(record)

            # Persist every finished deterministic batch immediately.
            records.extend(batch_records)
            write_jsonl(dst, records)

        _validate_existing_prefix(records, df, policy_name)
        if len(records) != len(df):
            raise RuntimeError(
                f"{policy_name} generation ended at {len(records)}/{len(df)} rows"
            )
        return records
    finally:
        clear_gpu(model)
        del tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument(
        "--policy",
        choices=POLICY_ORDER,
        help="Generate only one fixed policy. Default: all four.",
    )
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else cfg.get("safety_batch_size", 4)
    )
    policies = (args.policy,) if args.policy else POLICY_ORDER

    for name in policies:
        rows = generate_for_policy(
            cfg,
            name,
            batch_size=batch_size,
            resume=args.resume,
        )
        print(f"{name}: {len(rows)} deterministic responses", flush=True)


if __name__ == "__main__":
    main()

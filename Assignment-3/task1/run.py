import argparse
import json
import os
import shutil
from pathlib import Path
import yaml
from task1.protocol import validate_protocol


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("output_root", "result_root", "cache_dir"):
        config[key] = str(Path(config[key]).expanduser())
    manifest = Path(config["output_root"]) / "data" / "manifest.json"
    if manifest.exists():
        resolved = json.loads(manifest.read_text(encoding="utf-8")).get("model_revision")
        if resolved:
            if config.get("model_revision") and config["model_revision"] != resolved:
                raise ValueError("Configured model revision differs from prepared data manifest")
            config["model_revision"] = resolved
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["diagnose", "prepare", "preflight", "train", "evaluate", "weight",
                                          "activations", "qualitative", "aggregate", "export"])
    parser.add_argument("--config", default="configs/task1.yaml")
    parser.add_argument("--method", choices=["full", "l2sp", "lora4", "lora16", "oplora16"])
    parser.add_argument("--stop-update", type=int, choices=[312, 625, 937, 1250])
    parser.add_argument("--cpu-only", action="store_true", help="Run local static checks; GPU integration remains NOT RUN")
    parser.add_argument("--condition", choices=["baseline", "full_mid", "full_final", "l2sp_final",
                                                "lora4_final", "lora16_final", "oplora16_final"])
    args = parser.parse_args()
    config = read_config(args.config)
    validate_protocol(config)
    os.environ["HF_HOME"] = config["cache_dir"]
    os.environ["HF_DATASETS_CACHE"] = str(Path(config["cache_dir"]) / "datasets")
    Path(config["output_root"]).mkdir(parents=True, exist_ok=True)
    Path(config["result_root"]).mkdir(parents=True, exist_ok=True)
    if args.phase == "diagnose":
        from common.device import diagnostics
        print(json.dumps(diagnostics(config["output_root"]), indent=2))
    elif args.phase == "prepare":
        from task1.prepare_data import prepare
        from task1.diagnostics import create_probe_ids
        print(json.dumps(prepare(config), indent=2))
        print(json.dumps(create_probe_ids(config), indent=2))
    elif args.phase == "train":
        if not args.method:
            parser.error("train requires --method")
        from task1.training import run_training
        print(run_training(config, args.method, args.stop_update))
    elif args.phase == "preflight":
        from task1.preflight import run_preflight
        print(json.dumps(run_preflight(config, Path(__file__).resolve().parents[1] / "requirements.txt",
                                       cpu_only=args.cpu_only), indent=2, default=str))
    elif args.phase == "evaluate":
        if not args.condition:
            parser.error("evaluate requires --condition")
        from task1.evaluation import evaluate
        print(evaluate(config, args.condition))
    elif args.phase == "weight":
        from task1.diagnostics import run_weight_diagnostics
        print(run_weight_diagnostics(config))
    elif args.phase == "activations":
        from task1.diagnostics import run_activation_diagnostics
        print(f"Saved {len(run_activation_diagnostics(config))} activation records")
    elif args.phase == "qualitative":
        if not args.condition:
            parser.error("qualitative requires --condition")
        from task1.qualitative import generate
        print(f"Saved {len(generate(config, args.condition))} responses")
    elif args.phase == "aggregate":
        from task1.plots import aggregate
        from task1.qualitative import annotation_template, combine_outputs
        print(aggregate(config))
        if all((Path(config["result_root"]) / "qualitative" / f"{c}.json").exists()
               for c in ("baseline", "full_mid", "full_final", "l2sp_final", "lora4_final", "lora16_final", "oplora16_final")):
            print(combine_outputs(config))
            print(annotation_template(config))
    elif args.phase == "export":
        if not args.method:
            parser.error("export requires --method")
        source = Path(config["output_root"]) / args.method
        if not source.exists():
            raise FileNotFoundError(source)
        archive = Path(config["output_root"]) / f"{args.method}_export.zip"
        total = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
        if shutil.disk_usage(archive.parent).free < total * 1.05:
            raise RuntimeError("Insufficient free disk space to export full recovery checkpoints")
        shutil.make_archive(str(archive.with_suffix("")), "zip", source)
        print(archive)


if __name__ == "__main__":
    main()

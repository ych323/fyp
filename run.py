"""CLI entry point. All training/evaluation assets are loaded locally."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / "models" / "hf_cache"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "report", "all", "verify"])
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--model", choices=["mbert", "xlmr", "xlmr_dann"])
    args = parser.parse_args()
    from src.common import ROOT, experiment_dir, read_config
    config = read_config(args.config)
    if args.command == "all":
        # Sequential subprocesses release GPU memory between independent experiments.
        subprocess.run([sys.executable, "run.py", "prepare", "--config", args.config], cwd=ROOT, check=True)
        for model in ("mbert", "xlmr", "xlmr_dann"):
            out = experiment_dir(config) / model
            if not (out / "run_metadata.json").exists():
                subprocess.run([sys.executable, "run.py", "train", "--model", model, "--config", args.config], cwd=ROOT, check=True)
            if not (out / "metrics.json").exists():
                subprocess.run([sys.executable, "run.py", "evaluate", "--model", model, "--config", args.config], cwd=ROOT, check=True)
            subprocess.run([sys.executable, "run.py", "report", "--config", args.config], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "run.py", "verify", "--config", args.config], cwd=ROOT, check=True)
    elif args.command == "prepare":
        from src.prepare import prepare
        prepare(config)
    elif args.command in ("train", "evaluate"):
        if not args.model:
            parser.error("--model is required")
        if args.command == "train":
            from src.train import train
            train(config, args.model)
        else:
            from src.evaluate import evaluate
            evaluate(config, args.model)
    elif args.command == "report":
        from src.report import report
        report(config)
    else:
        from src.verify import verify
        verify(config)


if __name__ == "__main__":
    main()

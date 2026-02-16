#!/usr/bin/env python3
"""Run kvzip evaluations one at a time.

Usage:
    # Run all kvzip jobs (4 datasets × 3 ratios = 12 jobs):
    python scripts/run_kvzip.py

    # Run a single combo:
    python scripts/run_kvzip.py --dataset ruler --data_dir 4096 --compression_ratio 0.3

    # Specify GPU:
    python scripts/run_kvzip.py --gpu 2

    # Custom model:
    python scripts/run_kvzip.py --model /path/to/model
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "phase1_qwen3"
DEFAULT_MODEL = os.path.expanduser("~/models/Qwen3-8B")

DATASETS = [
    ("ruler", "4096"),
    ("ruler", "16384"),
    ("longbench-v2", ""),
    ("infinitebench", "longbook_qa_eng"),
]

COMPRESSION_RATIOS = [0.3, 0.5, 0.7]


def result_dir_name(dataset, data_dir, model, cr):
    model_tag = model.replace("/", "--")
    if data_dir:
        return f"{dataset}__{data_dir}__{model_tag}__kvzip__{cr:.2f}"
    else:
        return f"{dataset}__{model_tag}__kvzip__{cr:.2f}"


def run_one(dataset, data_dir, cr, model, output_dir, gpu, fraction=1.0):
    name = result_dir_name(dataset, data_dir, model, cr)
    result_path = output_dir / name / "metrics.json"

    if result_path.exists():
        print(f"[skip] {name} (already done)")
        return True

    # Clean incomplete results
    if (output_dir / name).exists():
        import shutil
        shutil.rmtree(output_dir / name)

    print(f"\n[gpu:{gpu}] {name}")

    kvpress_eval_dir = PROJECT_DIR.parent / "kvpress" / "evaluation"
    cmd = [
        sys.executable, str(PROJECT_DIR / "scripts" / "eval_wrapper.py"),
        "--config_file", "/dev/null",
        "--model", model,
        "--dataset", dataset,
        "--press_name", "kvzip",
        "--compression_ratio", str(cr),
        "--output_dir", str(output_dir),
        "--fraction", str(fraction),
    ]
    if data_dir:
        cmd += ["--data_dir", data_dir]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    result = subprocess.run(cmd, cwd=str(kvpress_eval_dir), env=env)

    if result.returncode != 0:
        print(f"[FAIL] {name}")
        return False
    else:
        print(f"[done] {name}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run kvzip evaluations")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", default=None, help="Run only this dataset")
    parser.add_argument("--data_dir", default=None, help="Data dir for single dataset run")
    parser.add_argument("--compression_ratio", type=float, default=None, help="Run only this CR")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to evaluate (e.g. 0.1 for 10%%)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single job mode
    if args.dataset and args.compression_ratio is not None:
        data_dir = args.data_dir or ""
        ok = run_one(args.dataset, data_dir, args.compression_ratio, args.model, output_dir, args.gpu, args.fraction)
        sys.exit(0 if ok else 1)

    # Run all 12 kvzip jobs
    done, failed, skipped = 0, 0, 0
    for ds, dd in DATASETS:
        for cr in COMPRESSION_RATIOS:
            name = result_dir_name(ds, dd, args.model, cr)
            if (output_dir / name / "metrics.json").exists():
                print(f"[skip] {name}")
                skipped += 1
                continue
            ok = run_one(ds, dd, cr, args.model, output_dir, args.gpu, args.fraction)
            if ok:
                done += 1
            else:
                failed += 1

    print(f"\n=== kvzip done: {done}  skipped: {skipped}  failed: {failed} ===")


if __name__ == "__main__":
    main()

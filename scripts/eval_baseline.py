#!/usr/bin/env python3
"""Evaluate base Qwen3-8B on RULER with SnapKV at various compression ratios.

Usage:
    python scripts/eval_baseline.py \
        --model_path /mnt/data/zichuanfu/models/Qwen3-8B \
        --kvpress_eval_dir ~/kvpress_repo/evaluation \
        --output_dir ~/SparseKV/analysis/ruler_results \
        --device cuda:0
"""
import sys
import os
import json
import argparse
import torch
import gc


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base model on RULER with SnapKV")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--kvpress_eval_dir", type=str, required=True, help="Path to kvpress/evaluation dir")
    parser.add_argument("--output_dir", type=str, default="./analysis/ruler_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--compression_ratios", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--context_length", type=int, default=4096, help="RULER context length")
    parser.add_argument("--fraction", type=float, default=0.1, help="Fraction of RULER data to use")
    parser.add_argument("--model_tag", type=str, default=None, help="Model tag for output dir naming")
    return parser.parse_args()


def load_model(model_path, device):
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model ready!")
    return model, tokenizer


def run_one_eval(model, tokenizer, model_tag, press_name, compression_ratio, output_dir, context_length, fraction):
    config = EvaluationConfig(
        dataset="ruler",
        data_dir=str(context_length),
        model=model_tag,
        press_name=press_name,
        compression_ratio=compression_ratio,
        fraction=fraction,
        output_dir=output_dir,
    )

    runner = EvaluationRunner(config)
    pipe = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)
    runner.pipeline = pipe

    out_dir = runner._setup_directories()
    results_dir = config.get_results_dir(out_dir)
    predictions_file = results_dir / "predictions.csv"
    metrics_file = results_dir / "metrics.json"
    config_file = results_dir / "config.yaml"

    if predictions_file.exists() and metrics_file.exists():
        print(f"SKIP (exists): {results_dir.name}")
        with open(metrics_file) as f:
            return json.load(f)

    runner._setup_press()
    runner._load_and_prepare_dataset()
    runner._run_inference()
    runner._save_results(predictions_file)
    runner._calculate_and_save_metrics(metrics_file)
    config.save_config(config_file)

    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return {"status": "completed"}


if __name__ == "__main__":
    args = parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Setup kvpress evaluation imports
    kvpress_eval_dir = os.path.expanduser(args.kvpress_eval_dir)
    sys.path.insert(0, kvpress_eval_dir)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvpress import KVPressTextGenerationPipeline
    from evaluate import EvaluationConfig, EvaluationRunner

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Derive model tag from path if not specified
    model_tag = args.model_tag or os.path.basename(args.model_path).replace("/", "--")

    model, tokenizer = load_model(args.model_path, args.device)

    results = {}
    for cr in args.compression_ratios:
        print(f"\n{'='*60}")
        print(f"{model_tag} + snapkv compression={cr}")
        print(f"{'='*60}")
        try:
            metrics = run_one_eval(
                model, tokenizer, model_tag, "snapkv", cr,
                output_dir, args.context_length, args.fraction,
            )
            results[f"snapkv_{cr}"] = metrics
            vals = [v['string_match'] for v in metrics.values() if isinstance(v, dict) and 'string_match' in v]
            if vals:
                print(f"  => avg string_match: {sum(vals)/len(vals):.2f}")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[f"snapkv_{cr}"] = {"error": str(e)}

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for key, metrics in results.items():
        if "error" in metrics:
            print(f"  {key}: ERROR - {metrics['error']}")
        else:
            vals = [v['string_match'] for v in metrics.values() if isinstance(v, dict) and 'string_match' in v]
            if vals:
                print(f"  {key}: avg={sum(vals)/len(vals):.2f}")
    print("\nALL DONE")

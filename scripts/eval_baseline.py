#!/usr/bin/env python3
"""Evaluate model on RULER with SnapKV at various compression ratios.

Supports base model or base + LoRA adapter (auto merge).
Supports parallel evaluation across multiple GPUs.

Usage:
    # Base model, all GPUs:
    python scripts/eval_baseline.py \
        --model_path /mnt/data/zichuanfu/models/Qwen3-8B \
        --kvpress_eval_dir ~/kvpress/evaluation

    # LoRA model:
    python scripts/eval_baseline.py \
        --model_path /mnt/data/zichuanfu/models/Qwen3-8B \
        --adapter_path ~/models/adapters/v7 \
        --model_tag trained_v7 \
        --kvpress_eval_dir ~/kvpress/evaluation

    # Single GPU:
    python scripts/eval_baseline.py \
        --model_path /mnt/data/zichuanfu/models/Qwen3-8B \
        --kvpress_eval_dir ~/kvpress/evaluation \
        --devices cuda:0
"""
import sys
import os
import json
import argparse
import torch
import gc
import multiprocessing as mp


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on RULER with SnapKV")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--kvpress_eval_dir", type=str, required=True, help="Path to kvpress/evaluation dir")
    parser.add_argument("--output_dir", type=str, default="./analysis/ruler_results", help="Output directory")
    parser.add_argument("--devices", type=str, nargs="+", default=None, help="GPU devices (default: all available)")
    parser.add_argument("--compression_ratios", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--context_length", type=int, default=4096, help="RULER context length")
    parser.add_argument("--fraction", type=float, default=0.1, help="Fraction of RULER data to use")
    parser.add_argument("--model_tag", type=str, default=None, help="Model tag for output dir naming")
    return parser.parse_args()


def run_single_cr(cr, model_path, adapter_path, kvpress_eval_dir, output_dir, device, context_length, fraction, model_tag):
    """Run evaluation for a single compression ratio on a specific GPU."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    sys.path.insert(0, kvpress_eval_dir)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvpress import KVPressTextGenerationPipeline
    from evaluate import EvaluationConfig, EvaluationRunner

    print(f"[{device}] Loading model for CR={cr}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )

    if adapter_path is not None:
        from peft import PeftModel
        print(f"[{device}] Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"[{device}] Merging LoRA...")
        model = model.merge_and_unload()

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    config = EvaluationConfig(
        dataset="ruler",
        data_dir=str(context_length),
        model=model_tag,
        press_name="snapkv",
        compression_ratio=cr,
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
        print(f"[{device}] SKIP (exists): CR={cr}")
        with open(metrics_file) as f:
            metrics = json.load(f)
    else:
        runner._setup_press()
        runner._load_and_prepare_dataset()
        runner._run_inference()
        runner._save_results(predictions_file)
        runner._calculate_and_save_metrics(metrics_file)
        config.save_config(config_file)
        with open(metrics_file) as f:
            metrics = json.load(f)

    vals = [v['string_match'] for v in metrics.values() if isinstance(v, dict) and 'string_match' in v]
    avg = sum(vals) / len(vals) if vals else 0
    print(f"[{device}] CR={cr} => avg string_match: {avg:.2f}")
    return cr, metrics


def worker(args_tuple):
    """Wrapper for multiprocessing."""
    try:
        return run_single_cr(*args_tuple)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return args_tuple[0], {"error": str(e)}


if __name__ == "__main__":
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    kvpress_eval_dir = os.path.expanduser(args.kvpress_eval_dir)
    output_dir = os.path.expanduser(args.output_dir)
    adapter_path = os.path.expanduser(args.adapter_path) if args.adapter_path else None
    os.makedirs(output_dir, exist_ok=True)

    model_tag = args.model_tag or os.path.basename(args.model_path).replace("/", "--")
    if args.devices is None:
        num_gpus = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(num_gpus)]
    else:
        devices = args.devices
    crs = args.compression_ratios

    print(f"Model: {args.model_path}")
    if adapter_path:
        print(f"Adapter: {adapter_path}")
    print(f"Tag: {model_tag}")
    print(f"Devices: {devices}")
    print(f"Compression ratios: {crs}")
    print(f"Context length: {args.context_length}")
    print()

    if len(devices) == 1:
        results = {}
        for cr in crs:
            cr_result, metrics = worker((
                cr, args.model_path, adapter_path, kvpress_eval_dir, output_dir,
                devices[0], args.context_length, args.fraction, model_tag,
            ))
            results[f"snapkv_{cr}"] = metrics
    else:
        task_args = []
        for i, cr in enumerate(crs):
            device = devices[i % len(devices)]
            task_args.append((
                cr, args.model_path, adapter_path, kvpress_eval_dir, output_dir,
                device, args.context_length, args.fraction, model_tag,
            ))

        num_workers = min(len(devices), len(crs))
        with mp.Pool(num_workers) as pool:
            pool_results = pool.map(worker, task_args)

        results = {}
        for cr, metrics in pool_results:
            results[f"snapkv_{cr}"] = metrics

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for key, metrics in sorted(results.items()):
        if "error" in metrics:
            print(f"  {key}: ERROR - {metrics['error']}")
        else:
            vals = [v['string_match'] for v in metrics.values() if isinstance(v, dict) and 'string_match' in v]
            if vals:
                print(f"  {key}: avg={sum(vals)/len(vals):.2f}")
    print("\nALL DONE")

#!/usr/bin/env python3
"""Wrapper around kvpress evaluate.py that records latency, throughput, and peak GPU memory.

Supports --model_tag to decouple local model path from result directory naming.
When provided, creates a local symlink so evaluate.py sees the model as "Qwen/Qwen3-8B"
(or whatever the tag maps to), producing correctly-named result directories.
"""

import subprocess
import sys
import time
import json
import os
import threading

def monitor_gpu_memory(interval=1.0, result={}):
    """Monitor peak GPU memory in background thread."""
    import subprocess
    result["peak_mb"] = 0
    result["running"] = True
    while result["running"]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                text=True
            )
            mem_values = [int(x.strip()) for x in out.strip().split("\n") if x.strip()]
            current_max = max(mem_values) if mem_values else 0
            result["peak_mb"] = max(result["peak_mb"], current_max)
        except Exception:
            pass
        time.sleep(interval)

def main():
    # Parse --model_tag (our custom arg, not passed to evaluate.py)
    eval_args = sys.argv[1:]
    model_tag = None
    filtered_args = []
    i = 0
    while i < len(eval_args):
        if eval_args[i] == "--model_tag" and i + 1 < len(eval_args):
            model_tag = eval_args[i + 1]
            i += 2
            continue
        filtered_args.append(eval_args[i])
        i += 1
    eval_args = filtered_args

    # Parse output_dir and model from args
    output_dir = None
    model_name = None
    model_arg_idx = None
    for i, arg in enumerate(eval_args):
        if arg == "--output_dir" and i + 1 < len(eval_args):
            output_dir = eval_args[i + 1]
        if arg == "--model" and i + 1 < len(eval_args):
            model_name = eval_args[i + 1]
            model_arg_idx = i + 1

    # If model_tag provided and model is a local path, create symlink so evaluate.py
    # sees a clean name like "Qwen/Qwen3-8B" instead of "/mnt/data/.../Qwen3-8B"
    symlink_model_path = None
    if model_tag and model_name and model_arg_idx is not None:
        # model_tag is like "Qwen--Qwen3-8B", convert to path: "Qwen/Qwen3-8B"
        rel_model_path = model_tag.replace("--", "/")
        real_model_path = os.path.expanduser(model_name)

        if os.path.isdir(real_model_path) and rel_model_path != real_model_path:
            # Create symlink in cwd: e.g. ./Qwen/Qwen3-8B -> /mnt/data/.../Qwen3-8B
            parts = rel_model_path.split("/")
            if len(parts) >= 2:
                symlink_dir = os.path.join(*parts[:-1])
                symlink_path = rel_model_path
                os.makedirs(symlink_dir, exist_ok=True)
                try:
                    os.symlink(real_model_path, symlink_path)
                except FileExistsError:
                    # Another process already created it, that's fine
                    pass
                symlink_model_path = symlink_path
                # Override --model in args to use the symlink
                eval_args[model_arg_idx] = symlink_path
                print(f"[eval_wrapper] Symlinked: {symlink_path} -> {real_model_path}")

    # Start GPU memory monitor
    mem_result = {}
    monitor_thread = threading.Thread(target=monitor_gpu_memory, args=(1.0, mem_result), daemon=True)
    monitor_thread.start()

    # Run evaluation
    start_time = time.time()

    cmd = [sys.executable, "evaluate.py"] + eval_args
    print(f"[eval_wrapper] Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    elapsed = time.time() - start_time
    mem_result["running"] = False
    monitor_thread.join(timeout=3)

    # Profiling data
    profile_data = {
        "total_time_seconds": round(elapsed, 2),
        "total_time_minutes": round(elapsed / 60, 2),
        "peak_gpu_memory_mb": mem_result.get("peak_mb", -1),
        "peak_gpu_memory_gb": round(mem_result.get("peak_mb", 0) / 1024, 2),
    }

    # Try to compute throughput from predictions
    if output_dir:
        try:
            subdirs = []
            for d in os.listdir(output_dir):
                pred_path = os.path.join(output_dir, d, "predictions.csv")
                if os.path.exists(pred_path):
                    subdirs.append((os.path.getmtime(pred_path), d))
            if subdirs:
                subdirs.sort(reverse=True)
                latest_dir = subdirs[0][1]
                pred_path = os.path.join(output_dir, latest_dir, "predictions.csv")

                with open(pred_path) as f:
                    n_samples = sum(1 for _ in f) - 1

                profile_data["num_samples"] = n_samples
                profile_data["seconds_per_sample"] = round(elapsed / max(n_samples, 1), 2)
                profile_data["samples_per_minute"] = round(n_samples / (elapsed / 60), 2)

                profile_path = os.path.join(output_dir, latest_dir, "profiling.json")
                with open(profile_path, "w") as f:
                    json.dump(profile_data, f, indent=2)
                print(f"[eval_wrapper] Profiling saved to {profile_path}")
        except Exception as e:
            print(f"[eval_wrapper] Warning: could not save profiling: {e}")

    # Print summary
    print(f"\n[eval_wrapper] === Profiling Summary ===")
    print(f"  Total time:       {profile_data['total_time_minutes']:.1f} min")
    print(f"  Peak GPU memory:  {profile_data.get('peak_gpu_memory_gb', '?')} GB")
    if 'samples_per_minute' in profile_data:
        print(f"  Throughput:       {profile_data['samples_per_minute']:.1f} samples/min")
        print(f"  Latency:          {profile_data['seconds_per_sample']:.2f} s/sample")

    sys.exit(result.returncode)

if __name__ == "__main__":
    main()

#!/bin/bash
# Run all evaluations on a multi-GPU server (no slurm).
# Each eval gets its own GPU, runs in parallel.
#
# Usage:
#   bash scripts/run_all.sh base         # Base Qwen3-8B
#   bash scripts/run_all.sh v6a          # v6a adapter
#   bash scripts/run_all.sh v6b          # v6b adapter
#   bash scripts/run_all.sh v7           # v7 adapter
#   bash scripts/run_all.sh all          # All of the above
#
# Environment variables:
#   MODEL_PATH    - path to base model (default: /mnt/data/zichuanfu/models/Qwen3-8B)
#   ADAPTER_DIR   - path to adapter directory (default: ~/models/adapters)
#   KVPRESS_EVAL  - path to kvpress/evaluation (default: ~/kvpress/evaluation)
#   OUTPUT_DIR    - output directory (default: ~/SparseKV/analysis/ruler_results)
#   GPUS          - comma-separated GPU ids (default: all available)
#   DATASETS      - override dataset list (default: ruler:4096)
#   LOG_DIR       - log directory (default: ~/eval_logs)

set -e

MODEL_PATH=${MODEL_PATH:-/mnt/data/zichuanfu/models/Qwen3-8B}
ADAPTER_DIR=${ADAPTER_DIR:-~/models/adapters}
KVPRESS_EVAL=${KVPRESS_EVAL:-~/kvpress/evaluation}
OUTPUT_DIR=${OUTPUT_DIR:-~/SparseKV/analysis/ruler_results}
LOG_DIR=${LOG_DIR:-~/eval_logs}
mkdir -p "$LOG_DIR"

# Auto-detect GPUs if not specified
if [ -z "$GPUS" ]; then
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
    GPUS=$(seq -s, 0 $((NUM_GPUS-1)))
fi
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}
echo "Using GPUs: ${GPUS} (${NUM_GPUS} total)"

# Evaluation configs
PRESSES=("snapkv:0.0" "snapkv:0.3" "snapkv:0.5" "snapkv:0.7" "snapkv:0.9")

run_model() {
    local MODEL_TAG=$1
    local ADAPTER_ARG=$2

    echo ""
    echo "============================================================"
    echo "  Evaluating: ${MODEL_TAG}"
    echo "============================================================"

    PIDS=()
    GPU_IDX=0

    for press_entry in "${PRESSES[@]}"; do
        PRESS="${press_entry%%:*}"
        CR="${press_entry##*:}"
        CR_FMT=$(printf "%.2f" "$CR")

        # Check if already done
        RESULT_DIR="${OUTPUT_DIR}/ruler__4096__${MODEL_TAG}__${PRESS}__${CR_FMT}__fraction0.100"
        if [ -f "${RESULT_DIR}/metrics.json" ]; then
            echo "  [skip] ${MODEL_TAG} ${PRESS} CR=${CR} (already done)"
            continue
        fi

        # Wait for a free GPU if all are busy
        while [ ${#PIDS[@]} -ge ${NUM_GPUS} ]; do
            NEW_PIDS=()
            for pid in "${PIDS[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    NEW_PIDS+=("$pid")
                fi
            done
            PIDS=("${NEW_PIDS[@]}")
            if [ ${#PIDS[@]} -ge ${NUM_GPUS} ]; then
                sleep 5
            fi
        done

        GPU=${GPU_LIST[$((GPU_IDX % NUM_GPUS))]}
        GPU_IDX=$((GPU_IDX + 1))

        LOG_FILE="${LOG_DIR}/${MODEL_TAG}_${PRESS}_${CR_FMT}.log"

        echo "  [gpu:${GPU}] ${MODEL_TAG} ${PRESS} CR=${CR} -> ${LOG_FILE}"

        python ~/SparseKV/scripts/eval_baseline.py \
            --model_path "${MODEL_PATH}" \
            ${ADAPTER_ARG} \
            --kvpress_eval_dir "${KVPRESS_EVAL}" \
            --output_dir "${OUTPUT_DIR}" \
            --model_tag "${MODEL_TAG}" \
            --devices "cuda:${GPU}" \
            --compression_ratios ${CR} \
            > "${LOG_FILE}" 2>&1 &

        PIDS+=($!)
    done

    # Wait for all jobs for this model
    echo "  Waiting for ${#PIDS[@]} remaining jobs..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" || echo "  [warn] PID $pid exited with error"
    done
    echo "  Done: ${MODEL_TAG}"
}

run_target() {
    case $1 in
        base)
            run_model "Qwen--Qwen3-8B" ""
            ;;
        v6a)
            run_model "trained_v6a" "--adapter_path ${ADAPTER_DIR}/v6a"
            ;;
        v6b)
            run_model "trained_v6b" "--adapter_path ${ADAPTER_DIR}/v6b"
            ;;
        v7)
            run_model "trained_v7" "--adapter_path ${ADAPTER_DIR}/v7"
            ;;
        all)
            run_target base
            run_target v6a
            run_target v6b
            run_target v7
            ;;
        *)
            echo "Unknown target: $1"
            echo "Usage: bash scripts/run_all.sh {base|v6a|v6b|v7|all}"
            exit 1
            ;;
    esac
}

TARGET=${1:-all}
run_target "$TARGET"

echo ""
echo "============================================================"
echo "  ALL EVALUATIONS COMPLETE"
echo "  Results: ${OUTPUT_DIR}"
echo "  Logs: ${LOG_DIR}"
echo "============================================================"

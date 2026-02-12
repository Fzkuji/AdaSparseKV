#!/bin/bash
# Run all baseline evaluations on a multi-GPU server (no slurm).
# Replaces submit_all.sh for non-slurm environments.
#
# Usage:
#   bash scripts/run_all.sh                    # Run all 52 combos
#   bash scripts/run_all.sh --model Qwen/Qwen3-8B   # Specify model
#
# Environment variables:
#   MODEL         - model name/path (default: Qwen/Qwen3-8B)
#   GPUS          - comma-separated GPU ids (default: all available)
#   OUTPUT_DIR    - output directory (default: ./results/phase1_qwen3)
#   LOG_DIR       - log directory (default: ~/eval_logs)
#   MAX_PARALLEL  - max parallel jobs (default: number of GPUs)

set -e

MODEL=${MODEL:-Qwen/Qwen3-8B}
OUTPUT_DIR=${OUTPUT_DIR:-./results/phase1_qwen3}
LOG_DIR=${LOG_DIR:-~/eval_logs}
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Auto-detect GPUs
if [ -z "$GPUS" ]; then
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
    GPUS=$(seq -s, 0 $((NUM_GPUS-1)))
fi
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}
MAX_PARALLEL=${MAX_PARALLEL:-$NUM_GPUS}

echo "Model:        ${MODEL}"
echo "Output:       ${OUTPUT_DIR}"
echo "GPUs:         ${GPUS} (${NUM_GPUS} total, max ${MAX_PARALLEL} parallel)"
echo "Logs:         ${LOG_DIR}"
echo ""

DATASETS=("ruler:4096" "ruler:16384" "longbench-v2:" "infinitebench:")
PRESSES=("no_press:0" "snapkv:0.3" "snapkv:0.5" "snapkv:0.7" "streaming_llm:0.3" "streaming_llm:0.5" "streaming_llm:0.7" "critical_snapkv:0.3" "critical_snapkv:0.5" "critical_snapkv:0.7" "kvzip:0.3" "kvzip:0.5" "kvzip:0.7")

TOTAL=$((${#DATASETS[@]} * ${#PRESSES[@]}))
echo "Total combos: ${TOTAL}"
echo ""

# Change to kvpress evaluation dir
cd ~/kvpress/evaluation

PIDS=()
GPU_IDX=0
RUNNING=0
SKIPPED=0
LAUNCHED=0

for ds_entry in "${DATASETS[@]}"; do
    DS_NAME="${ds_entry%%:*}"
    DS_DIR="${ds_entry##*:}"

    for press_entry in "${PRESSES[@]}"; do
        PRESS="${press_entry%%:*}"
        CR="${press_entry##*:}"
        CR_FMT=$(printf "%.2f" "$CR")

        JOB_NAME="${DS_NAME}_${DS_DIR:-default}_${PRESS}_${CR_FMT}"

        # Build data_dir arg
        DATA_DIR_ARG=""
        if [ -n "$DS_DIR" ]; then
            DATA_DIR_ARG="--data_dir $DS_DIR"
        fi

        # Check if already done
        RESULT_NAME="${DS_NAME}__${DS_DIR:-4096}__$(echo $MODEL | sed 's|/|--|g')__${PRESS}__${CR_FMT}"
        RESULT_PATH="${OUTPUT_DIR}/${RESULT_NAME}/metrics.json"
        if [ -f "$RESULT_PATH" ]; then
            echo "[skip] ${JOB_NAME} (done)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Wait for a free slot
        while [ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]; do
            NEW_PIDS=()
            for pid in "${PIDS[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    NEW_PIDS+=("$pid")
                fi
            done
            PIDS=("${NEW_PIDS[@]}")
            if [ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]; then
                sleep 5
            fi
        done

        GPU=${GPU_LIST[$((GPU_IDX % NUM_GPUS))]}
        GPU_IDX=$((GPU_IDX + 1))

        LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"

        echo "[gpu:${GPU}] ${JOB_NAME} -> ${LOG_FILE}"

        CUDA_VISIBLE_DEVICES="${GPU}" python ~/SparseKV/scripts/eval_wrapper.py \
            --model "${MODEL}" \
            --dataset "${DS_NAME}" ${DATA_DIR_ARG} \
            --press_name "${PRESS}" \
            --compression_ratio "${CR}" \
            --output_dir "${OUTPUT_DIR}" \
            > "${LOG_FILE}" 2>&1 &

        PIDS+=($!)
        LAUNCHED=$((LAUNCHED + 1))
    done
done

# Wait for all remaining jobs
echo ""
echo "Waiting for ${#PIDS[@]} remaining jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "[warn] PID $pid exited with error"
done

echo ""
echo "============================================================"
echo "  COMPLETE"
echo "  Launched: ${LAUNCHED}  Skipped: ${SKIPPED}  Total: ${TOTAL}"
echo "  Results: ${OUTPUT_DIR}"
echo "  Logs: ${LOG_DIR}"
echo "============================================================"

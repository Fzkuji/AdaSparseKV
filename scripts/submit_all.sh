#!/bin/bash
# 提交所有 baseline 评测任务
# 用法: bash scripts/submit_all.sh [model_key]
#   model_key: qwen3 (默认), llama, gpt-oss
set -e

MODEL_KEY=${1:-qwen3}

case $MODEL_KEY in
    qwen3)   MODEL="Qwen/Qwen3-8B" ;;
    llama)   MODEL="meta-llama/Llama-3.1-8B-Instruct" ;;
    gpt-oss) MODEL="openai/gpt-oss-20b" ;;
    *)       MODEL="$MODEL_KEY" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="./results/phase1_${MODEL_KEY}"

DATASETS=("ruler:4096" "ruler:16384" "longbench:" "aime25:")
PRESSES=("no_press:0" "snapkv:0.3" "snapkv:0.5" "snapkv:0.7" "streaming_llm:0.3" "streaming_llm:0.5" "streaming_llm:0.7" "critical_snapkv:0.3" "critical_snapkv:0.5" "critical_snapkv:0.7" "kvzip:0.3" "kvzip:0.5" "kvzip:0.7")

echo "Model: $MODEL"
echo "Will submit ${#DATASETS[@]} x ${#PRESSES[@]} = $(( ${#DATASETS[@]} * ${#PRESSES[@]} )) jobs"
echo ""

COUNT=0
for ds_entry in "${DATASETS[@]}"; do
    DS_NAME="${ds_entry%%:*}"
    DS_DIR="${ds_entry##*:}"
    
    for press_entry in "${PRESSES[@]}"; do
        PRESS="${press_entry%%:*}"
        CR="${press_entry##*:}"
        
        JOB_NAME="${MODEL_KEY}_${DS_NAME}_${PRESS}_${CR}"
        
        # 构建 data_dir 参数
        DATA_DIR_ARG=""
        if [ -n "$DS_DIR" ]; then
            DATA_DIR_ARG="--data_dir $DS_DIR"
        fi

        cat > /tmp/job_${JOB_NAME}.sh << HEREDOC
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=/home/zichuanfu2/logs/output_%j.txt
#SBATCH --error=/home/zichuanfu2/logs/error_%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=23:00:00

conda activate adasparse

cd ~/kvpress/evaluation

CUDA_VISIBLE_DEVICES="0,1" python evaluate.py \\
    --model ${MODEL} \\
    --dataset ${DS_NAME} ${DATA_DIR_ARG} \\
    --press_name ${PRESS} \\
    --compression_ratio ${CR} \\
    \\
    --output_dir ${OUTPUT_DIR}
HEREDOC

        echo "  [$COUNT] $JOB_NAME"
        sbatch /tmp/job_${JOB_NAME}.sh
        COUNT=$((COUNT + 1))
    done
done

echo ""
echo "Submitted $COUNT jobs. Check: squeue -u zichuanfu2"

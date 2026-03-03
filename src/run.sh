#!/usr/bin/env bash
set -euo pipefail

##############################################################
# 阿里百炼 Agent 竞赛 —— 一键推理脚本
# 用法：
#   bash run.sh [OUTPUT_PATH] [ROLL_OUT_COUNT]
#
# 参数说明：
#   OUTPUT_PATH     预测结果基础输出目录（默认：../output），每次运行在其下生成带时间戳子目录
#   ROLL_OUT_COUNT  每道题推理轮数（默认：1，竞赛推荐使用 1）
##############################################################

######################################
### 0. 加载 .env 配置               ###
######################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"

if [ -f "$ENV_FILE" ]; then
    echo "==== Loading environment variables from $ENV_FILE ===="
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Warning: .env file not found at $ENV_FILE"
    echo "Please create it with DASHSCOPE_API_KEY and other settings."
fi

######################################
### 1. 参数配置                     ###
######################################

# 生成本次运行的唯一标识：模型名_时间戳
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MODEL="${DASHSCOPE_MAIN_MODEL:-qwen-max}"
MODEL_TAG="$(echo "$MODEL" | tr '/' '-')"   # 将模型名中的斜杠替换为短横线，避免路径问题
RUN_ID="${MODEL_TAG}_${RUN_TIMESTAMP}"

# 竞赛数据集路径（固定）
DATA_FILEPATH="${SCRIPT_DIR}/../data/question.jsonl"

# 基础输出目录，可由第一个参数指定
BASE_OUTPUT="${1:-${SCRIPT_DIR}/../output}"

# 本次运行专属子目录：output/{MODEL_TAG}_{TIMESTAMP}/
OUTPUT_PATH="${BASE_OUTPUT}/${RUN_ID}"

# 推理轮数：竞赛默认 1 轮，可由第二个参数指定
ROLL_OUT_COUNT="${2:-1}"

# 并发工作线程数
MAX_WORKERS="${MAX_WORKERS:-1}"

# 全局汇总文件（追加写入，方便跨轮次比较）
SUMMARY_PATH="${BASE_OUTPUT}/summary.jsonl"

echo "==== Configuration ===="
echo "  Run ID       : $RUN_ID"
echo "  Data file    : $DATA_FILEPATH"
echo "  Output dir   : $OUTPUT_PATH"
echo "  Summary file : $SUMMARY_PATH"
echo "  Model        : $MODEL"
echo "  Rollout count: $ROLL_OUT_COUNT"
echo "  Max workers  : $MAX_WORKERS"
echo "======================="

# 检查数据文件是否存在
if [ ! -f "$DATA_FILEPATH" ]; then
    echo "Error: Data file not found at $DATA_FILEPATH"
    exit 1
fi

mkdir -p "$OUTPUT_PATH"

######################################
### 2. 启动推理                     ###
######################################

echo "==== Starting inference... ===="

uv run python -u run_multi_react.py \
    --model        "$MODEL" \
    --data_filepath "$DATA_FILEPATH" \
    --output       "$OUTPUT_PATH" \
    --roll_out_count "$ROLL_OUT_COUNT" \
    --max_workers  "$MAX_WORKERS" 

echo "==== Inference completed! ===="

######################################
### 3. 统计输出（可选）             ###
######################################
# 注：竞赛数据集无标准答案，此处只计算过程统计（工具调用次数等），跳过 LLM 评判。
# 如需对有答案的验证集进行完整评估，手动执行：
#   python evaluate.py --input_folder <PRED_DIR> --dataset <DATASET> --restore_result_path summary.jsonl

# MODEL_NAME=$(echo "$MODEL" | tr '/' '-')
# PREDICTION_PATH="${OUTPUT_PATH}/${MODEL_NAME}/competition"

# if [ -f "${PREDICTION_PATH}/iter1.jsonl" ]; then
#     echo "==== Computing inference statistics... ===="
#     uv run python evaluate.py \
#         --input_folder  "$PREDICTION_PATH" \
#         --dataset       "competition" \
#         --run_id        "$RUN_ID" \
#         --restore_result_path "$SUMMARY_PATH" \
#         --skip_judge
#     echo "Statistics saved to ${SUMMARY_PATH}"
# else
#     echo "No prediction files found in $PREDICTION_PATH, skipping statistics."
# fi


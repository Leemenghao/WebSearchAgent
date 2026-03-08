#!/usr/bin/env bash
set -euo pipefail

##############################################################
# 阿里百炼 Agent 竞赛 —— 一键推理脚本
# 用法：
#   bash run.sh [OUTPUT_PATH] [ROLL_OUT_COUNT]
#
# 参数说明：
#   OUTPUT_PATH       预测结果基础输出目录（默认：../output），每次运行在其下生成带时间戳子目录
#   ROLL_OUT_COUNT    每道题推理轮数（默认：1，竞赛推荐使用 1）
#   RESUME_OUTPUT_DIR 【续跑模式】指定已有的输出目录，跳过时间戳生成直接追加写入
#   RESUME_DATA_FILE  【续跑模式】指定自定义数据文件路径（配合 RESUME_OUTPUT_DIR 使用）
#
# 续跑示例：
#   bash run.sh ../output 1 ../output/qwen3-max_20260303_233627 ../output/qwen3-max_20260303_233627/qwen3-max/competition/remaining_41.jsonl
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
# DATA_FILEPATH="${SCRIPT_DIR}/../data/question.jsonl"
DATA_FILEPATH="${SCRIPT_DIR}/../data/test.jsonl"

# 基础输出目录，可由第一个参数指定
BASE_OUTPUT="${1:-${SCRIPT_DIR}/../output}"

# 本次运行专属子目录：output/{MODEL_TAG}_{TIMESTAMP}/
# 若第 3 个参数指定了已有目录，则进入续跑模式，直接使用该目录
if [ -n "${3:-}" ]; then
    OUTPUT_PATH="${3}"
    echo "==== Resume mode: using existing output dir ===="
else
    OUTPUT_PATH="${BASE_OUTPUT}/${RUN_ID}"
fi

# 若第 4 个参数指定了自定义数据文件，则覆盖默认数据路径
if [ -n "${4:-}" ]; then
    DATA_FILEPATH="${4}"
fi

# 推理轮数：竞赛默认 1 轮，可由第二个参数指定
ROLL_OUT_COUNT="${2:-1}"

# 并发工作线程数
MAX_WORKERS="${MAX_WORKERS:-1}"

echo "==== Configuration ===="
echo "  Run ID       : $RUN_ID"
echo "  Data file    : $DATA_FILEPATH"
echo "  Output dir   : $OUTPUT_PATH"
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

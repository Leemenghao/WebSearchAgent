# WebSearch Agent

基于 **ReAct 架构** 的 WebSearch Agent。

## 架构概览

```
问题
 └─► Decomposer（问题分解）─► Checker（计划校验）
                                      │
                                      ▼
                            MultiTurnReactAgent（主推理循环）
                            ├── search（Google Serper 批量搜索）
                            ├── visit（Jina 网页抓取）
                            └── Scratchpad（每3次工具调用更新摘要黑板）
                                      │
                                      ▼
                                  <answer>
```


## 快速开始

### 环境要求

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/)（包管理）

### 1. 克隆 & 安装依赖

```bash
git clone https://github.com/Leemenghao/WebSearchAgent.git
cd WebSearchAgent

uv sync
```

### 2. 配置环境变量

```bash
cp .env-example .env
```

编辑 `.env`，填入你的 API Key：

```ini
DASHSCOPE_API_KEY=sk-xxx        # 阿里百炼 API Key（必填）
GOOGLE_SEARCH_KEY=xxx           # Serper API Key（必填）
JINA_API_KEY=xxx                # Jina Reader API Key（必填）
```

其余参数（模型选择、最大调用轮数等）均有默认值，按需修改。

### 3. 准备数据

将题目文件放到 `data/` 目录，格式为 JSONL，每行一个对象：

```jsonl
{"id": 0, "question": "..."}
{"id": 1, "question": "..."}
```

> 默认读取 `data/question.jsonl`，可在 `src/run.sh` 中修改 `DATA_FILEPATH`。

### 4. 运行推理

```bash
cd src
bash run.sh
```

结果会输出到 `output/{model}_{timestamp}/` 目录下。

**可选参数：**

```bash
# 指定输出目录和推理轮数
bash run.sh ../output 1

# 续跑模式（跳过已完成题目）
bash run.sh ../output 1 ../output/qwen3-max_20260303_233627 ../output/qwen3-max_20260303_233627/qwen3-max/competition/remaining_41.jsonl
```

### 5. 提取提交结果

```bash
cd utils
uv run python extract_submit.py \
    --pred ../output/<run_dir>/<model>/competition/iter1.jsonl \
    --question ../data/question.jsonl \
    --output ../submit
```

输出的 JSONL 格式符合竞赛要求：

```jsonl
{"id": 0, "answer": "..."}
{"id": 1, "answer": "..."}
```

## 目录结构

```
ali_agent_2602/
├── src/
│   ├── react_agent.py       # 主 Agent 推理逻辑
│   ├── prompt.py            # 所有 Prompt 定义
│   ├── tool_search.py       # Google Serper 搜索工具
│   ├── tool_visit.py        # Jina 网页抓取工具
│   ├── run_multi_react.py   # 多线程推理入口
│   └── run.sh               # 一键推理脚本
├── utils/
│   ├── extract_submit.py    # 提取答案生成提交文件
│   └── run.sh               # utils 工具脚本
├── data/
│   └── question.jsonl       # 竞赛题目
├── output/                  # 推理结果（自动生成）
├── submit/                  # 提交文件
├── .env-example             # 环境变量模板
├── pyproject.toml
└── uv.lock
```

## 参考

- [DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) — 参考架构

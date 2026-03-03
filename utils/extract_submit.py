"""
extract_submit.py
将推理输出 JSONL 按比赛题目顺序提取答案，生成满足提交要求的 JSONL 文件。

提交格式（每行一个 JSON）：
    {"id": 0, "answer": "xxx"}
    {"id": 1, "answer": "xxx"}
    ...

答案归一化规则（与比赛评测保持一致）：
    - 去除首尾空格
    - 转小写
    - 若为纯数字（含小数），转为整数字符串
    - 若预测为空，answer 字段为空字符串

用法：
    python extract_submit.py \
        --pred  <推理输出 iter1.jsonl 路径> \
        --question <比赛题目 question.jsonl 路径> \
        --output <输出提交文件路径>

    若不传参数，则使用下方 DEFAULT_* 默认值。
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ─── 默认路径 ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent   # ali_agent_2602/
DEFAULT_QUESTION = str(_ROOT / "data" / "question.jsonl")
DEFAULT_PRED = ""   # 需由命令行传入或修改此处
DEFAULT_OUTPUT = ""  # 默认与 --pred 同目录，文件名 submit.jsonl
# ──────────────────────────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    """比赛评测对齐的标准预处理：
    1. 去除首尾空格
    2. 转小写
    3. 纯数字（含小数）取整数
    4. 多实体：逗号或分号后补齐一个空格
    """
    t = text.strip().lower()
    # 纯数字（含小数）-> 整数
    if re.fullmatch(r"-?\d+(\.\d+)?", t):
        try:
            t = str(int(float(t)))
        except ValueError:
            pass
        return t
    # 多实体：逗号 / 分号后确保恰好一个空格
    t = re.sub(r'([,;])\s*', r'\1 ', t)
    t = t.strip()
    return t


def load_questions(path: str) -> list[tuple[int, str]]:
    """读取比赛题目，返回有序 (id, question) 列表。
    优先使用题目文件中的 id 字段，不存在时回退到行号（0-indexed）。
    """
    questions = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("id", lineno)          # 优先读取显式 id
            q = obj.get("question", "").strip()
            questions.append((qid, q))
    return questions


def load_predictions(path: str) -> dict[str, str]:
    """读取推理输出，返回 {question_text: prediction} 字典"""
    preds: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("question", "").strip()
            pred = obj.get("prediction", "").strip()
            if q:
                # 同一题目可能有多 rollout，后者覆盖前者（也可改为保留第一条）
                preds[q] = pred
    return preds


def extract(question_path: str, pred_path: str, output_path: str) -> None:
    questions = load_questions(question_path)   # list[(id, question)]
    preds = load_predictions(pred_path)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    missing_ids = []
    with open(output_path, "w", encoding="utf-8") as fout:
        for qid, q in questions:
            pred = preds.get(q, "")
            if pred == "":
                missing_ids.append(qid)
            answer = normalize(pred)  # 始终执行归一化
            record = {"id": qid, "answer": answer}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(questions)
    matched = total - len(missing_ids)
    print(f"[extract_submit] 题目总数: {total}")
    print(f"[extract_submit] 已匹配: {matched}")
    if missing_ids:
        print(f"[extract_submit] 未匹配（answer 为空）的 id: {missing_ids}")
    print(f"[extract_submit] 输出文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="提取推理结果并生成比赛提交 JSONL")
    parser.add_argument("--pred", default=DEFAULT_PRED,
                        help="推理输出 JSONL 路径（iter1.jsonl）")
    parser.add_argument("--question", default=DEFAULT_QUESTION,
                        help="比赛题目 JSONL 路径（默认 data/question.jsonl）")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="输出提交文件路径（默认与 pred 同目录）")
    args = parser.parse_args()

    if not args.pred:
        print("错误：请通过 --pred 指定推理输出文件路径", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.pred):
        print(f"错误：推理输出文件不存在: {args.pred}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.question):
        print(f"错误：题目文件不存在: {args.question}", file=sys.stderr)
        sys.exit(1)

    # output 未指定时，默认与 pred 同目录
    output_path = args.output or str(Path(args.pred).parent / "submit.jsonl")

    extract(
        question_path=args.question,
        pred_path=args.pred,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

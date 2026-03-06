"""
使用大模型对 submit.jsonl 的答案做“格式对齐”，不做重新作答。

功能：
- 输入：submit.jsonl（每行 {"id": int, "answer": str}）
- 读取原始问题（默认 data/question.jsonl）
- 调用大模型，仅基于“原答案 + 原问题”做格式化
- 输出到 submit 同目录（默认 submit_formatted.jsonl）

示例：
uv run python utils/format_submit_with_llm.py \
  --submit /home/lmh/lee/competition/ali_agent_2602/output/qwen3-max_20260302_020344/qwen3-max/competition/submit.jsonl
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass


SYSTEM_PROMPT = """You are a strict answer formatter for QA competition submissions.

Task: format the given raw answer using the original question as style constraints.
You MUST NOT solve the question and MUST NOT add new facts.
Only transform existing raw answer text into required output format.

Rules:
1) Keep semantics from raw answer only; do not infer new entities.
2) Output must be plain text only (no explanation, no markdown, no quotes).
3) If raw answer is empty or cannot be safely normalized, output empty string.
4) If answer is numeric, output integer only.
5) If multiple entities, separate with ", " exactly.
6) Match answer language to question language when possible by formatting only.
7) Remove leading phrases like "答案是", "final decision", etc.
8) If raw answer is long analysis, extract only its final concise answer span.
"""


def load_questions(question_path: Path) -> Dict[int, str]:
    qmap: Dict[int, str] = {}
    with question_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("id", i)
            qmap[int(qid)] = obj.get("question", "")
    return qmap


def load_submit(submit_path: Path) -> List[dict]:
    rows = []
    with submit_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def clean_output(text: str) -> str:
    if not text:
        return ""
    out = text.strip().strip('"').strip("'")
    lower = out.lower()
    if lower in {"n/a", "none", "null", "unknown", "无法确定", "无法判断", ""}:
        return ""
    if out.startswith("<answer>") and out.endswith("</answer>"):
        out = out[len("<answer>"):-len("</answer>")].strip()
    return out


def format_one(client: OpenAI, model: str, qid: int, question: str, raw_answer: str, max_retries: int = 3) -> str:
    if not raw_answer or not raw_answer.strip():
        return ""

    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Raw answer (must not be re-answered):\n"
        f"{raw_answer}\n\n"
        "Return ONLY the normalized final answer text."
    )

    for i in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=128,
                extra_body={"enable_thinking": False},
                timeout=60,
            )
            content = resp.choices[0].message.content or ""
            return clean_output(content)
        except Exception as e:
            if i == max_retries - 1:
                print(f"[format] id={qid} failed: {e}")
                return clean_output(raw_answer)
            time.sleep(1.0 * (i + 1))

    return clean_output(raw_answer)


def main():
    parser = argparse.ArgumentParser(description="Format submit.jsonl answers with LLM (format only, no re-answer)")
    parser.add_argument("--submit", required=True, help="Path to submit.jsonl")
    parser.add_argument("--question", default="/home/lmh/lee/competition/ali_agent_2602/data/question.jsonl", help="Path to question.jsonl")
    parser.add_argument("--output", default="", help="Output file path; default: <submit_dir>/submit_formatted.jsonl")
    parser.add_argument("--model", default=os.getenv("FORMATTER_MODEL", os.getenv("CHECKER_MODEL", "qwen3.5-flash")), help="Formatter model")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    submit_path = Path(args.submit).resolve()
    question_path = Path(args.question).resolve()
    output_path = Path(args.output).resolve() if args.output else submit_path.with_name("submit_formatted.jsonl")

    if not submit_path.exists():
        raise FileNotFoundError(f"submit not found: {submit_path}")
    if not question_path.exists():
        raise FileNotFoundError(f"question file not found: {question_path}")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set DASHSCOPE_API_KEY in environment or .env")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    qmap = load_questions(question_path)
    rows = load_submit(submit_path)

    results = [None] * len(rows)

    def worker(idx: int, row: dict):
        qid = int(row.get("id", idx))
        question = qmap.get(qid, "")
        raw_answer = str(row.get("answer", ""))
        formatted = format_one(client, args.model, qid, question, raw_answer)
        return idx, {"id": qid, "answer": formatted}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(worker, i, row) for i, row in enumerate(rows)]
        for fut in as_completed(futures):
            idx, rec = fut.result()
            results[idx] = rec

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] input : {submit_path}")
    print(f"[done] output: {output_path}")
    print(f"[done] model : {args.model}")


if __name__ == "__main__":
    main()

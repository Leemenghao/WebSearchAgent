"""
classify_questions.py
对 question.jsonl 中每道题的"期望答案类型"进行分类并统计。
用法：
    cd /home/lmh/lee/competition/ali_agent_2602/utils
    uv run python classify_questions.py [--workers 5] [--output result.jsonl]
"""

import json
import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

CLIENT = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = os.getenv("CHECKER_MODEL", "qwen3.5-flash")  # 快速轻量模型即可

SYSTEM_PROMPT = """你是一个问答系统分析专家。
你的任务：阅读问题，判断该问题期望的答案属于哪种类型。

请从以下类别中选择最合适的**一个**：

1. 人名 - 期望回答某人的姓名（真实人物）
2. 机构/组织名 - 公司、政府机构、协会、学校等
3. 地名 - 城市、国家、地区、建筑等地理位置
4. 年份/日期 - 某年、某月某日等时间点
5. 数字/数量 - 金额、次数、人数、距离等纯数值
6. 作品名 - 书籍、电影、电视剧、游戏、歌曲、论文等
7. 专有术语 - 科学术语、技术名词、学科概念、指标名等
8. 事件名 - 历史事件、比赛、运动等专有事件
9. 其他具体名词 - 物种学名、菜品、设备等不属于以上的具体名词
10. 是/否或选择 - 答案是是/否或从有限选项中选一个

只输出类别名称，不加任何解释，不加类别的数字。
例子：
期望的输出：人名
错误的输出：1. 人名
"""

def classify_one(item: dict) -> dict:
    question = item.get("question", "")
    try:
        resp = CLIENT.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            max_tokens=20,
            temperature=0,
            extra_body={"enable_thinking": False},
        )
        category = resp.choices[0].message.content.strip()
    except Exception as e:
        category = f"[ERROR] {e}"
    return {"id": item.get("id"), "question": question[:80], "category": category}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/question.jsonl")
    parser.add_argument("--output", default="/home/lmh/lee/competition/ali_agent_2602/data/classify_result.jsonl")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    input_path = os.path.join(os.path.dirname(__file__), args.input)
    output_path = os.path.join(os.path.dirname(__file__), args.output)

    with open(input_path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]

    print(f"共 {len(items)} 道题，使用模型 {MODEL}，并发 {args.workers} 线程...\n")

    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(classify_one, item): i for i, item in enumerate(items)}
        for done_idx, future in enumerate(as_completed(future_map), 1):
            idx = future_map[future]
            results[idx] = future.result()
            print(f"[{done_idx:3d}/{len(items)}] id={results[idx]['id']:>2}  "
                  f"类别={results[idx]['category']:<12}  "
                  f"{results[idx]['question']}", flush=True)

    # 写出详细结果
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    counter = Counter(r["category"] for r in results)
    total = len(results)

    print("\n" + "=" * 55)
    print(f"{'答案类别':<18} {'题数':>6}  {'占比':>7}")
    print("-" * 55)
    for cat, cnt in counter.most_common():
        bar = "█" * int(cnt / total * 30)
        print(f"{cat:<18} {cnt:>6}  {cnt/total:>6.1%}  {bar}")
    print("=" * 55)
    print(f"{'合计':<18} {total:>6}")
    print(f"\n详细结果已写入 → {output_path}")


if __name__ == "__main__":
    main()

"""
analyze_termination.py
分析 iter1.jsonl 中各题的 termination 字段，辅助诊断 Agent 优化方向。

直接修改下方 ITER1_PATH 指定目标文件，运行后在同目录输出：
  - termination_analysis.txt   人类可读的详细报告
  - termination_errors.jsonl   所有未正常结束的条目（便于进一步处理）

termination 类型说明：
  answer                                  正常结束，模型给出了答案
  generate an answer as token limit reached  Token 超限后强制给出答案
  format error: generate an answer...     Token 超限后强制回答但格式错误
  answer not found                        未触发 answer 标签，到达轮次上限
  exceed available llm calls              LLM 调用次数耗尽，模型未能输出答案
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# ─── 配置：修改此路径指向目标文件 ─────────────────────────────────────────────
ITER1_PATH = Path(
    "/home/lmh/lee/competition/ali_agent_2602/output"
    "/qwen3-max_20260302_020344/qwen3-max/competition/iter1.jsonl"
)
# ──────────────────────────────────────────────────────────────────────────────

# termination 是否视为"成功作答"
SUCCESS_TERMS = {"answer", "generate an answer as token limit reached"}


def count_tool_calls(messages: list) -> dict:
    """统计各工具被调用次数及是否存在连续重复调用（死循环）。

    注意：开启 ENABLE_THINKING=true 时，react_agent.py 会将 reasoning_content
    包装为 <think>...</think> 存入消息，thinking 内容中自然含有 JSON 字面量，
    不应视为实际工具调用，此处只统计 <tool_call> 标签内的真实调用。
    """
    tool_counts: Counter = Counter()
    prev_call = None
    max_consecutive = 0
    cur_consecutive = 1

    for msg in messages:
        if msg["role"] != "assistant":
            continue
        content = msg.get("content", "")
        # 只提取 <tool_call> 块，忽略 <think> 内的推理文本
        calls = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)

        for call_str in calls:
            try:
                call = json.loads(call_str.strip())
                tool_counts[call.get("name", "unknown")] += 1
                # 检测连续重复
                call_key = json.dumps(call, sort_keys=True)
                if call_key == prev_call:
                    cur_consecutive += 1
                    max_consecutive = max(max_consecutive, cur_consecutive)
                else:
                    cur_consecutive = 1
                prev_call = call_key
            except Exception:
                pass

    total_effective_calls = sum(tool_counts.values())
    return {
        "tool_counts": dict(tool_counts),
        "max_consecutive_same_call": max_consecutive,
        "total_effective_calls": total_effective_calls,
    }


def analyze(path: Path) -> None:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    term_counter: Counter = Counter(r.get("termination", "") for r in records)
    success = sum(term_counter[t] for t in SUCCESS_TERMS)
    failed = total - success

    # 按 termination 分组
    groups: dict = defaultdict(list)
    for r in records:
        groups[r.get("termination", "")].append(r)

    # ── 输出路径 ────────────────────────────────────────────────────────────────
    out_dir = path.parent
    txt_out = out_dir / "termination_analysis.txt"
    err_out = out_dir / "termination_errors.jsonl"

    lines = []
    lines.append("=" * 70)
    lines.append(f"  Termination Analysis: {path.name}")
    lines.append("=" * 70)
    lines.append(f"\n总题数: {total}  |  成功作答: {success}  |  未作答: {failed}\n")
    lines.append(f"{'Termination':<45} {'数量':>4}  {'占比':>6}")
    lines.append("-" * 60)
    for term, cnt in term_counter.most_common():
        flag = "✓" if term in SUCCESS_TERMS else "✗"
        lines.append(f"{flag} {term:<43} {cnt:>4}  {cnt/total*100:>5.1f}%")
    lines.append("")

    # ── 详细分析每个失败类型 ──────────────────────────────────────────────────
    for term, items in sorted(groups.items(), key=lambda x: -len(x[1])):
        if term in SUCCESS_TERMS:
            continue
        lines.append("=" * 70)
        lines.append(f"✗ [{term}]  ({len(items)} 题)")
        lines.append("=" * 70)
        for item in items:
            q = item.get("question", "")
            pred = item.get("prediction", "")
            msgs = item.get("messages", [])
            tc = count_tool_calls(msgs)

            lines.append(f"\n  ID/rollout: {item.get('rollout_id', '?')}")
            lines.append(f"  问题: {q[:100]}{'...' if len(q)>100 else ''}")
            lines.append(f"  预测: {pred[:80] if pred else '(空)'}")
            lines.append(f"  消息轮次: {len(msgs)}")
            if tc["tool_counts"]:
                tool_str = ", ".join(
                    f"{k}×{v}" for k, v in tc["tool_counts"].items()
                )
                lines.append(f"  工具调用: {tool_str}")
            if tc["max_consecutive_same_call"] > 2:
                lines.append(
                    f"  ⚠ 连续重复调用同一 tool {tc['max_consecutive_same_call']} 次（疑似死循环）"
                )
            # 消息轮次多但实际工具调用很少，说明模型大量思考却未行动
            if len(msgs) > 8 and tc["total_effective_calls"] == 0:
                lines.append(
                    f"  ⚠ 共 {len(msgs)} 条消息但无任何有效工具调用（模型全程未行动）"
                )
            elif len(msgs) > 12 and tc["total_effective_calls"] < 3:
                lines.append(
                    f"  ⚠ 共 {len(msgs)} 条消息仅 {tc['total_effective_calls']} 次有效工具调用"
                    "（工具调用稀少，可能陷入纯思考循环或 max_llm_calls 不足）"
                )
        lines.append("")

    # ── 成功作答中预测为空的异常 ────────────────────────────────────────────
    empty_preds = [
        r for r in records
        if r.get("termination") in SUCCESS_TERMS and not r.get("prediction", "").strip()
    ]
    if empty_preds:
        lines.append("=" * 70)
        lines.append(f"⚠ 成功终止但预测为空的题 ({len(empty_preds)} 题)")
        lines.append("=" * 70)
        for r in empty_preds:
            lines.append(f"  问题: {r.get('question','')[:100]}")

    report = "\n".join(lines)
    print(report)

    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n[analyze_termination] 报告已写入: {txt_out}")

    # ── 导出所有失败条目（去掉 messages 以减小体积）─────────────────────────
    with open(err_out, "w", encoding="utf-8") as f:
        for r in records:
            if r.get("termination", "") not in SUCCESS_TERMS:
                slim = {k: v for k, v in r.items() if k != "messages"}
                f.write(json.dumps(slim, ensure_ascii=False) + "\n")
    print(f"[analyze_termination] 失败条目已写入: {err_out}")


if __name__ == "__main__":
    analyze(ITER1_PATH)

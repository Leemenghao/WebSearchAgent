"""
test_think_mode.py

测试 Qwen think 模式下，模型是否会在 <think> 标签内嵌套其他标签
（如 <tool_call>、<answer>）。

分两轮测试：
  Round 1：首次提问，观察模型如何规划并发起工具调用
  Round 2：注入 tool_response，观察模型如何给出最终答案

运行前确保 .env 中 ENABLE_THINKING=true
"""

import os
import re
import sys

sys.path.insert(0, "../src")
from dotenv import load_dotenv
load_dotenv("../.env")

# 强制开启 think 模式（即使 .env 未设置）
os.environ["ENABLE_THINKING"] = "true"

from openai import OpenAI
from prompt import SYSTEM_PROMPT_MULTI, USER_PROMPT

# ─── 直接复用 react_agent 的 call_server 逻辑 ────────────────────────────────
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL = os.getenv("DASHSCOPE_MAIN_MODEL", "qwen3-max")
STOP_SEQUENCES = ["\n<tool_response>", "<tool_response>"]


def call_server(msgs: list) -> tuple[str, str]:
    """返回 (thinking_content, response_content)"""
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model=MODEL,
        messages=msgs,
        extra_body={"enable_thinking": True},
        stream=True,
        temperature=0.6,
        top_p=0.95,
    )
    thinking, response = "", ""
    stopped = False
    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            thinking += delta.reasoning_content
        if hasattr(delta, "content") and delta.content:
            response += delta.content
            for stop in STOP_SEQUENCES:
                pos = response.find(stop)
                if pos != -1:
                    response = response[:pos]
                    stopped = True
                    break
            if stopped:
                break
    return thinking, response


def analyze_nesting(thinking: str, response: str) -> None:
    """检测 <think> 内部是否包含 <tool_call> / <answer> 等标签"""
    tags_to_check = ["tool_call", "answer", "tool_response"]
    print("\n─── 嵌套标签检测 ───────────────────────────────────────")
    found_any = False
    for tag in tags_to_check:
        pattern = rf"<{tag}[\s>]"
        if re.search(pattern, thinking, re.IGNORECASE):
            print(f"  ⚠️  <think> 内部发现 <{tag}> 标签！")
            found_any = True
    if not found_any:
        print("  ✓  <think> 内部未发现其他标签嵌套，结构正常")

    # 检查 response 中 <think> 的位置（response 不应包含 <think>）
    if "<think>" in response:
        print("  ⚠️  response 部分包含 <think> 标签（预期不应出现）")
    else:
        print("  ✓  response 部分未包含 <think> 标签")


def print_section(title: str, content: str, max_chars: int = 600) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)
    if len(content) > max_chars:
        print(content[:max_chars] + f"\n  ... （共 {len(content)} 字符，已截断）")
    else:
        print(content)


# ─── 主测试流程 ───────────────────────────────────────────────────────────────
question = "法国的首都是哪里？"   # 简单问题，便于观察结构

messages = [
    {"role": "system", "content": SYSTEM_PROMPT_MULTI},
    {"role": "user",   "content": USER_PROMPT + question},
]

print(f"测试模型: {MODEL}")
print(f"测试问题: {question}")

# ── Round 1 ──────────────────────────────────────────────────────────────────
print("\n" + "█"*60)
print("  Round 1: 首次提问")
print("█"*60)

thinking1, response1 = call_server(messages)
print_section("thinking（reasoning_content）", thinking1)
print_section("response（content）", response1)
analyze_nesting(thinking1, response1)

# 判断模型行为
if "<tool_call>" in response1 and "</tool_call>" in response1:
    print("\n→ 模型发起了工具调用，进入 Round 2")
    messages.append({"role": "assistant", "content": response1.strip()})

    # 注入一个 mock tool_response
    mock_tool_response = "<tool_response>\nParis\n</tool_response>"
    messages.append({"role": "user", "content": mock_tool_response})

    # ── Round 2 ──────────────────────────────────────────────────────────────
    print("\n" + "█"*60)
    print("  Round 2: 注入 tool_response 后的回答")
    print("█"*60)

    thinking2, response2 = call_server(messages)
    print_section("thinking（reasoning_content）", thinking2)
    print_section("response（content）", response2)
    analyze_nesting(thinking2, response2)

    if "<answer>" in response2 and "</answer>" in response2:
        answer = response2.split("<answer>")[1].split("</answer>")[0].strip()
        print(f"\n✓ 最终答案: {answer}")
    else:
        print("\n⚠️  未检测到 <answer> 标签")

elif "<answer>" in response1:
    answer = response1.split("<answer>")[1].split("</answer>")[0].strip()
    print(f"\n→ 模型直接给出答案（无工具调用）: {answer}")
else:
    print("\n→ 模型既未调用工具也未给出 <answer>，输出结构异常")

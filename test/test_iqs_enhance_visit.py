"""
测试 IQSEnhancedVisit 工具：
  1. 环境变量检查（IQS_API_KEY）
  2. iqs_readpage() 轻量页（example.com）- 验证速度
  3. iqs_readpage() 真实内容页（Wikipedia）
  4. call() 单 URL 完整流程（抓取 + LLM 提取）
  5. call() 多 URL 并发
  6. 错误处理 — 无效 URL
  7. 参数格式错误
"""

import sys
import os
import time

sys.path.insert(0, "../src")

from dotenv import load_dotenv
load_dotenv("../.env")

from toll_iqs_enhance_visit import IQSEnhancedVisit, IQS_BASIC_URL

TOOL = IQSEnhancedVisit()
SEP = "─" * 60
PASS = "[OK]  "
FAIL = "[FAIL]"
WARN = "[WARN]"


def section(title):
    print(f"\n{'='*60}\n{title}\n{SEP}")


# ── Test 1: 环境变量 ──────────────────────────────────────────────────────────
section("Test 1: 环境变量检查")
api_key = os.getenv("IQS_API_KEY", "")
print(f"  IQS_API_KEY      : {api_key[:8]}...{api_key[-4:] if api_key else '(未设置)'}")
print(f"  IQS_BASIC_URL    : {IQS_BASIC_URL}")
print(f"  DASHSCOPE_API_KEY: {os.getenv('DASHSCOPE_API_KEY','(未设置)')[:8]}...")
if not api_key:
    print(f"  {FAIL} IQS_API_KEY 未设置，后续抓取测试将失败")
else:
    print(f"  {PASS} 环境变量就绪")


# ── Test 2: 轻量页速度测试 ────────────────────────────────────────────────────
section("Test 2: iqs_readpage() 轻量页（example.com）")
t0 = time.time()
raw = TOOL.iqs_readpage("https://www.example.com")
elapsed = time.time() - t0

if raw.startswith("[visit]"):
    print(f"  {FAIL} 抓取失败: {raw}  (耗时 {elapsed:.1f}s)")
else:
    print(f"  {PASS} 抓取成功，内容长度: {len(raw)} 字符，耗时: {elapsed:.1f}s")
    print("  " + raw[:200].replace("\n", "\n  "))


# ── Test 3: 真实内容页 ────────────────────────────────────────────────────────
section("Test 3: iqs_readpage() 真实内容页（Wikipedia）")
WIKI_URL = "https://en.wikipedia.org/wiki/Alibaba_Group"
print(f"  URL: {WIKI_URL}")
t0 = time.time()
raw3 = TOOL.iqs_readpage(WIKI_URL)
elapsed = time.time() - t0

if raw3.startswith("[visit]"):
    print(f"  {WARN} 抓取失败: {raw3}  (耗时 {elapsed:.1f}s)")
else:
    print(f"  {PASS} 抓取成功，内容长度: {len(raw3)} 字符，耗时: {elapsed:.1f}s")
    print("  前 400 字符:")
    print("  " + raw3[:400].replace("\n", "\n  "))


# ── Test 4: 完整流程（抓取 + LLM 提取）────────────────────────────────────────
section("Test 4: call() 单 URL 完整流程")
print(f"  URL : {WIKI_URL}")
GOAL = "When was Alibaba Group founded and who is the founder?"
print(f"  Goal: {GOAL}")

t0 = time.time()
result = TOOL.call({"url": WIKI_URL, "goal": GOAL})
elapsed = time.time() - t0

print(f"  耗时: {elapsed:.1f}s，返回长度: {len(result)} 字符")
has_ev = "Evidence in page:" in result
has_su = "Summary:" in result
print(f"  {PASS if has_ev else FAIL} 含 Evidence 字段: {has_ev}")
print(f"  {PASS if has_su else FAIL} 含 Summary  字段: {has_su}")
print("  " + result[:500].replace("\n", "\n  "))


# ── Test 5: 多 URL 并发 ───────────────────────────────────────────────────────
section("Test 5: call() 多 URL 并发")
URLS = ["https://www.example.com", "https://www.iana.org/domains/reserved"]
print(f"  URLs: {URLS}")
t0 = time.time()
multi = TOOL.call({"url": URLS, "goal": "What is this website for?"})
elapsed = time.time() - t0
parts = [p for p in multi.split("=======") if p.strip()]
print(f"  耗时: {elapsed:.1f}s，共 {len(parts)} 段，总长度: {len(multi)} 字符")
print(f"  {PASS if len(parts) >= 2 else WARN} 预期 2 段，实际 {len(parts)} 段")
for i, p in enumerate(parts, 1):
    print(f"\n  --- 段落 {i} ---\n  " + p.strip()[:200].replace("\n", "\n  "))


# ── Test 6: 无效 URL ──────────────────────────────────────────────────────────
section("Test 6: 错误处理 — 无效 URL")
BAD_URL = "https://this-domain-does-not-exist-xyzzy99.com/page"
print(f"  URL: {BAD_URL}")
t0 = time.time()
err_result = TOOL.call({"url": BAD_URL, "goal": "测试"})
elapsed = time.time() - t0
graceful = any(k in err_result for k in ["could not be accessed", "could not be processed", "Failed"])
print(f"  耗时: {elapsed:.1f}s")
print(f"  {PASS if graceful else WARN} 错误{'优雅处理' if graceful else '处理异常'}: {err_result[:120]}")


# ── Test 7: 参数错误 ──────────────────────────────────────────────────────────
section("Test 7: 参数格式错误")
bad = TOOL.call({"wrong": "value"})
ok = bad.startswith("[Visit] Invalid")
print(f"  {PASS if ok else WARN} {bad}")


print(f"\n{'='*60}\n所有测试完成\n{'='*60}")
import os
import time

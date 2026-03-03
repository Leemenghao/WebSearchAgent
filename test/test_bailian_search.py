import sys
sys.path.insert(0, "../src")

from dotenv import load_dotenv
load_dotenv("../.env")

from tool_bailian_search import bailian_search, BailianSearch

# 1. 单次搜索
print("=== 单次搜索 ===")
print(bailian_search("2026年春节是哪天", count=3))

# 2. 工具类 call（多 query）
print("\n=== 批量搜索（工具类） ===")
tool = BailianSearch()
result = tool.call({"query": ["Python asyncio", "阿里云百炼API"]})
print(result[:800])

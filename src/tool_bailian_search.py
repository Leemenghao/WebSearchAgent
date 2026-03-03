import json
import os
import threading
import time
from typing import List, Union

import requests
from qwen_agent.tools.base import BaseTool, register_tool

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
_MCP_URL = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"
_DEFAULT_COUNT = 10

# 全局限速：最多 5 QPS，防止 20 个并发线程打爆配额
_rate_lock = threading.Lock()
_last_call_time = 0.0
_MIN_INTERVAL = 0.25  # 秒，对应 4 QPS


def bailian_search(query: str, count: int = _DEFAULT_COUNT) -> str:
    global _last_call_time
    # 全局限速：串行占用锁，确保两次调用间隔 >= _MIN_INTERVAL
    with _rate_lock:
        now = time.time()
        gap = now - _last_call_time
        if gap < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - gap)
        _last_call_time = time.time()

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "bailian_web_search",
            "arguments": {"query": query, "count": count},
        },
    }

    max_retries = 6
    result = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(_MCP_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:
                wait = 2.0 * (2 ** attempt)
                print(f"[bailian_search] HTTP 429, retry {attempt+1}/{max_retries} after {wait:.1f}s ...")
                time.sleep(wait)
                continue
            result = resp.json()
        except Exception as e:
            print(f"[bailian_search] Request error (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                return f"Bailian search timeout for '{query}', please try again later."
            time.sleep(2)
            continue

        # 检查 JSON-RPC 层错误
        if result.get("error"):
            return f"Bailian search error: {result['error'].get('message', result['error'])}"

        tool_result = result.get("result", {})
        if tool_result.get("isError"):
            try:
                err_body = json.loads(tool_result["content"][0]["text"])
                app_status = err_body.get("status", 0)
            except Exception:
                app_status = 0
            if app_status == 429:
                wait = 2.0 * (2 ** attempt)
                print(f"[bailian_search] App-level 429, retry {attempt+1}/{max_retries} after {wait:.1f}s ...")
                time.sleep(wait)
                continue
            # 其他 isError
            print(f"[bailian_search] isError for '{query}': {str(tool_result)[:200]}")
            return f"Bailian search returned an error for '{query}'."

        # 成功
        break
    else:
        return f"Bailian search failed for '{query}' after {max_retries} retries (persistent 429)."

    try:
        pages = json.loads(tool_result["content"][0]["text"]).get("pages", [])
    except Exception:
        return f"Failed to parse search results for '{query}'."

    if not pages:
        return f"No results found for '{query}'. Try a more general query."

    snippets = []
    for idx, page in enumerate(pages, 1):
        entry = f"{idx}. [{page.get('title', '')}]({page.get('url', '')})"
        hostname = page.get("hostname", "")
        if hostname and hostname != "无":
            entry += f"\nSource: {hostname}"
        if page.get("snippet"):
            entry += f"\n{page['snippet']}"
        snippets.append(entry)

    return (
        f"A Bailian web search for '{query}' found {len(snippets)} results:\n\n"
        f"## Web Results\n" + "\n\n".join(snippets)
    )


@register_tool("search", allow_overwrite=True)
class BailianSearch(BaseTool):
    name = "search"
    description = (
        "Performs batched web searches via Alibaba Bailian real-time search: "
        "supply an array 'query'; the tool retrieves the top 10 results per query in one call."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings. Include multiple complementary search queries in a single call.",
            },
        },
        "required": ["query"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        assert DASHSCOPE_API_KEY, "Please set the DASHSCOPE_API_KEY environment variable."
        try:
            query = params["query"]
        except Exception:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        if isinstance(query, str):
            return bailian_search(query)

        assert isinstance(query, List)
        results = []
        for q in query:
            results.append(bailian_search(q))
            time.sleep(0.5)  # 批量查询时每条间隔 0.5s，减少 429
        return "\n=======\n".join(results)

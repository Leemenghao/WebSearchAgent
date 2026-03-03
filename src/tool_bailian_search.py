import json
import os
import time
from typing import List, Union

import requests
from qwen_agent.tools.base import BaseTool, register_tool

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
_MCP_URL = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"
_DEFAULT_COUNT = 10


def bailian_search(query: str, count: int = _DEFAULT_COUNT) -> str:
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
    for attempt in range(max_retries):
        try:
            resp = requests.post(_MCP_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:
                wait = 0.5 * (2 ** attempt)
                print(f"[bailian_search] 429 Rate limit, retry {attempt+1}/{max_retries} after {wait:.1f}s ...")
                time.sleep(wait)
                continue
            result = resp.json()
            break
        except Exception as e:
            print(f"[bailian_search] Request error (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                return f"Bailian search timeout for '{query}', please try again later."
            time.sleep(1)
    else:
        return f"Bailian search failed for '{query}' after {max_retries} retries."

    # 解析 JSON-RPC 响应
    if result.get("error"):
        return f"Bailian search error: {result['error'].get('message', result['error'])}"

    tool_result = result.get("result", {})
    if tool_result.get("isError"):
        return f"Bailian search returned an error for '{query}'."

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
            time.sleep(0.2)
        return "\n=======\n".join(results)

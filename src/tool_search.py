from qwen_agent.tools.base import BaseTool, register_tool
import json
from typing import List, Union
import requests
import os
import time
import threading

GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY")
SERPER_MIN_INTERVAL = float(os.getenv("SERPER_MIN_INTERVAL", "0.25"))

_rate_lock = threading.Lock()
_last_call_time = 0.0


def _throttle_serper_calls():
    global _last_call_time
    with _rate_lock:
        now = time.time()
        gap = now - _last_call_time
        if gap < SERPER_MIN_INTERVAL:
            time.sleep(SERPER_MIN_INTERVAL - gap)
        _last_call_time = time.time()


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {
                    "type": "string"
                    },
                    "description": "Array of query strings. Include multiple complementary search queries in a single call."
                },
            },
        "required": ["query"],
    }

    def google_search(self, query: str):
        url = 'https://google.serper.dev/search'
        headers = {
            'X-API-KEY': GOOGLE_SEARCH_KEY,
            'Content-Type': 'application/json',
        }
        data = {
            "q": query,
            "num": 10,
            "extendParams": {
                "country": "en",
                "page": 1,
            },
        }

        max_retries = 8
        for i in range(max_retries):
            try:
                _throttle_serper_calls()
                response = requests.post(url, headers=headers, data=json.dumps(data), timeout=15)
                if response.status_code == 429:
                    # 指数退避：0.5s, 1s, 2s, 4s, 8s ...
                    wait = 0.5 * (2 ** i)
                    print(f"[search] 429 Rate limit, retry {i+1}/{max_retries} after {wait:.1f}s ...")
                    time.sleep(wait)
                    continue
                results = response.json()
                break
            except Exception as e:
                print(f"[search] Request/parse error for query '{query}' (attempt {i+1}/{max_retries}): {e}")
                if i == max_retries - 1:
                    return f"Google search Timeout, return None, Please try again later."
                time.sleep(1)
        else:
            return f"Google search failed after {max_retries} retries (rate limit)."

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        try:
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            visit_recommended = []   # 截断/过短 snippet 的 URL，建议 visit
            idx = 0
            # 取前10条（按 Google 综合排序返回，通常相关性优先）
            for page in results["organic"][:10]:
                idx += 1
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet_text = page.get("snippet", "")
                snippet = "\n" + snippet_text if snippet_text else ""

                # 标记 snippet 截断或过短的结果
                is_truncated = snippet_text.endswith("...") or snippet_text.endswith("…")
                is_short = len(snippet_text) < 120
                if is_truncated or is_short:
                    visit_flag = " ⚠️[snippet truncated, visit recommended]"
                    visit_recommended.append(f"#{idx} {page['link']}")
                else:
                    visit_flag = ""

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}{snippet}{visit_flag}"
                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)

            content = f"A Google search for '{query}' found {len(web_snippets)} results (top 10 shown):\n\n## Web Results\n" + "\n\n".join(web_snippets)

            if visit_recommended:
                content += (
                    f"\n\n💡 **Snippets are truncated or too short for these results — you MUST call `visit` on at least one of them to get the full answer:**\n"
                    + "\n".join(visit_recommended)
                )

            return content
        except Exception as e:
            print(f"[search] Formatting error for query '{query}': {e}")
            return f"No results found for '{query}'. Try with a more general query, or remove the year filter."


    def call(self, params: Union[str, dict], **kwargs) -> str:
        assert GOOGLE_SEARCH_KEY is not None, "Please set the GOOGLE_SEARCH_KEY environment variable."
        try:
            query = params["query"]
        except Exception:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            response = self.google_search(query)
        else:
            assert isinstance(query, List)
            results = []
            for q in query:
                results.append(self.google_search(q))
                time.sleep(0.3)  # 串行执行，约 3 req/s，低于 5 req/s 限制
            response = "\n=======\n".join(results)
        return response

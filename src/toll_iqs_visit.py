"""
IQS 网页解析工具
使用阿里云 IQS ReadPageBasic HTTP 接口（直连，无需 SDK）替换 Jina Reader。

环境变量:
    IQS_API_KEY             - IQS API Key（X-API-Key）
    DASHSCOPE_API_KEY       - DashScope API Key（用于 LLM 内容提取）
    DASHSCOPE_SUMMARY_MODEL - 摘要模型（默认 qwen3-max）
    WEBCONTENT_MAXLENGTH    - 网页内容最大长度（默认 150000）
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union

import requests
from openai import OpenAI
from qwen_agent.tools.base import BaseTool, register_tool

from prompt import EXTRACTOR_PROMPT

# ── 配置 ─────────────────────────────────────────────────────────────────────
IQS_BASIC_URL = "https://cloud-iqs.aliyuncs.com/readpage/basic"
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))


# ── Tool 注册 ─────────────────────────────────────────────────────────────────
@register_tool("visit", allow_overwrite=True)
class IQSEnhancedVisit(BaseTool):
    """通过 IQS ReadPageBasic HTTP 接口访问网页，使用 LLM 提取与目标相关的内容。"""

    name = "visit"
    description = "Visit webpage(s) and return the summary of the content."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "minItems": 1,
                "description": (
                    "The URL(s) of the webpage(s) to visit. "
                    "Can be a single URL or an array of URLs."
                ),
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s).",
            },
        },
        "required": ["url", "goal"],
    }

    # ── 入口 ─────────────────────────────────────────────────────────────────
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except Exception:
            return (
                "[Visit] Invalid request format: "
                "Input must be a JSON object containing 'url' and 'goal' fields"
            )

        if isinstance(url, str):
            response = self.readpage(url, goal)
        else:
            assert isinstance(url, List)
            results: list[str] = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self.readpage, u, goal): u for u in url}
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        results.append(f"Error fetching {futures[future]}: {e}")
            response = "\n=======\n".join(results)

        print(f"[iqs_visit] Summary Length={len(response)}")
        return response.strip()

    # ── LLM 摘要调用 ──────────────────────────────────────────────────────────
    def call_server(self, msgs: list, max_tries: int = 10) -> str:
        """调用 DashScope LLM 对网页内容进行结构化提取。"""
        api_key = os.getenv("DASHSCOPE_API_KEY", "EMPTY")
        api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        summary_model = os.getenv("DASHSCOPE_SUMMARY_MODEL", "qwen3-max")

        client = OpenAI(api_key=api_key, base_url=api_base)
        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(
                    model=summary_model,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=0.7,
                )
                content = chat_response.choices[0].message.content
                if content:
                    # 尝试从非纯 JSON 回复中提取 JSON 块
                    try:
                        json.loads(content)
                    except json.JSONDecodeError:
                        left = content.find("{")
                        right = content.rfind("}")
                        if left != -1 and right != -1 and left <= right:
                            content = content[left : right + 1]
                    return content
            except Exception:
                if attempt == max_tries - 1:
                    return ""
                continue
        return ""

    # ── IQS HTTP 抓取 ──────────────────────────────────────────────────────────
    def iqs_readpage(self, url: str) -> str:
        """
        调用 IQS ReadPageBasic HTTP 接口抓取网页内容。
        返回 markdown 文本；失败时返回以 "[visit]" 开头的错误消息。
        """
        api_key = os.getenv("IQS_API_KEY", "")
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        }
        payload = {
            "url": url,
            "maxAge": 0,
            "formats": ["markdown", "text"],
            "readability": {
                "readabilityMode": "normal",  # 剔除页头/页脚/导航噪音
                "excludeAllImages": True,
            },
            "timeout": 30000,
        }
        max_retries = 3

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    IQS_BASIC_URL,
                    headers=headers,
                    json=payload,
                    timeout=35,
                )

                if resp.status_code == 429:
                    wait = 2 ** attempt
                    print(f"[iqs_visit] 429 rate limit, waiting {wait}s ...")
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    print(f"[iqs_visit] {url} → HTTP {resp.status_code}: {resp.text[:200]}")
                    return "[visit] Failed to read page."

                data = resp.json().get("data", {})
                site_status = data.get("statusCode", 200)

                if site_status != 200:
                    err = data.get("errorMessage", "")
                    print(f"[iqs_visit] {url} → site status={site_status}, error={err}")
                    if site_status == 4290:   # 目标站限流，重试
                        time.sleep(2 ** attempt)
                        continue
                    return "[visit] Failed to read page."

                content = data.get("markdown") or data.get("text")
                return content if content else "[visit] Empty content."

            except requests.exceptions.Timeout:
                print(f"[iqs_visit] Timeout attempt {attempt + 1}/{max_retries} for {url}")
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."
            except Exception as e:
                print(f"[iqs_visit] Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."

        return "[visit] Failed to read page."

    # ── 主流程：抓取 + LLM 提取 ───────────────────────────────────────────────
    def readpage(self, url: str, goal: str) -> str:
        """
        抓取网页并用 LLM 提取与 goal 相关的关键信息。

        返回格式化的 Evidence + Summary 字符串。
        """
        max_fetch_attempts = 3

        for fetch_attempt in range(max_fetch_attempts):
            raw_content = self.iqs_readpage(url)

            # ── 获取到有效内容 ────────────────────────────────────────────────
            if raw_content and not raw_content.startswith("[visit]"):
                raw_content = raw_content[:WEBCONTENT_MAXLENGTH]
                messages = [
                    {
                        "role": "user",
                        "content": EXTRACTOR_PROMPT.format(
                            webpage_content=raw_content, goal=goal
                        ),
                    }
                ]
                extracted = self.call_server(messages)

                # 内容过长导致 LLM 返回为空 → 逐步截断重试
                summary_retries = 3
                while len(extracted) < 10 and summary_retries >= 0:
                    if summary_retries > 0:
                        truncate_length = int(0.7 * len(raw_content))
                        print(
                            f"[iqs_visit] Truncating content: "
                            f"{len(raw_content)} → {truncate_length} chars "
                            f"(retry {3 - summary_retries + 1}/3)"
                        )
                    else:
                        truncate_length = 25000
                        print(
                            f"[iqs_visit] Final truncation to {truncate_length} chars"
                        )
                    raw_content = raw_content[:truncate_length]
                    messages = [
                        {
                            "role": "user",
                            "content": EXTRACTOR_PROMPT.format(
                                webpage_content=raw_content, goal=goal
                            ),
                        }
                    ]
                    extracted = self.call_server(messages)
                    summary_retries -= 1

                # ── 解析 LLM 返回的 JSON ──────────────────────────────────────
                parse_retries = 0
                while parse_retries < 3:
                    try:
                        extracted = json.loads(extracted)
                        break
                    except (json.JSONDecodeError, TypeError):
                        extracted = self.call_server(messages)
                        parse_retries += 1

                if parse_retries >= 3 or not isinstance(extracted, dict):
                    return (
                        f"The useful information in {url} for user goal '{goal}' as follows:\n\n"
                        "Evidence in page:\n"
                        "The provided webpage content could not be accessed. "
                        "Please check the URL or file format.\n\n"
                        "Summary:\n"
                        "The webpage content could not be processed, "
                        "and therefore, no information is available.\n\n"
                    )

                return (
                    f"The useful information in {url} for user goal '{goal}' as follows:\n\n"
                    f"Evidence in page:\n{extracted.get('evidence', '')}\n\n"
                    f"Summary:\n{extracted.get('summary', '')}\n\n"
                )

            # ── 最后一次获取仍然失败 ──────────────────────────────────────────
            if fetch_attempt == max_fetch_attempts - 1:
                return (
                    f"The useful information in {url} for user goal '{goal}' as follows:\n\n"
                    "Evidence in page:\n"
                    "The provided webpage content could not be accessed. "
                    "Please check the URL or file format.\n\n"
                    "Summary:\n"
                    "The webpage content could not be processed, "
                    "and therefore, no information is available.\n\n"
                )

        return "[visit] Failed to read page."

import json
import os
from typing import Dict, List, Optional, Union
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI
import tiktoken
from transformers import AutoTokenizer 
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool


MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 40))
MAX_TOKEN_LENGTH = int(os.getenv('MAX_LENGTH', 128 * 1024 - 500))

print(f'Running with MAX_LLM_CALL_PER_RUN = {MAX_LLM_CALL_PER_RUN}')

class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description,
                         files=files,
                         **kwargs)
        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        self.model_type = llm.get("model_type", "")

    def call_server(self, msgs, model: str, max_tries=10):
        # 使用阿里百炼（DashScope）兼容 OpenAI 接口
        api_key = os.getenv("DASHSCOPE_API_KEY", "EMPTY")
        api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        enable_thinking = os.getenv("ENABLE_THINKING", "false").lower() == "true"

        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

        STOP_SEQUENCES = ["\n<tool_response>", "<tool_response>"]

        for attempt in range(max_tries):
            try:
                if enable_thinking:
                    # 流式调用，支持思考模式（reasoning_content）
                    completion = client.chat.completions.create(
                        model=model,
                        messages=msgs,
                        extra_body={"enable_thinking": True},
                        stream=True,
                        temperature=self.llm_generate_cfg.get('temperature', 0.6),
                        top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    )
                    thinking_content = ""
                    response_content = ""
                    stopped = False
                    for chunk in completion:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        # 累积思考过程
                        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                            thinking_content += delta.reasoning_content
                        # 累积响应内容，手动检查停止序列
                        if hasattr(delta, "content") and delta.content:
                            response_content += delta.content
                            for stop_seq in STOP_SEQUENCES:
                                pos = response_content.find(stop_seq)
                                if pos != -1:
                                    response_content = response_content[:pos]
                                    stopped = True
                                    break
                            if stopped:
                                break
                    # 将思考过程包裹在 <think> 标签中，与回复内容合并
                    if thinking_content:
                        content = f"<think>{thinking_content}</think>\n{response_content}"
                    else:
                        content = response_content
                else:
                    # 普通非流式调用
                    chat_response = client.chat.completions.create(
                        model=model,
                        messages=msgs,
                        stop=["\n<tool_response>", "<tool_response>"],
                        temperature=self.llm_generate_cfg.get('temperature', 0.6),
                        top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    )
                    content = chat_response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                if attempt == (max_tries - 1):
                    print(f"DashScope API error: {e}")
                    return f"DashScope API error"
                continue

        return "DashScope API empty response"

    def count_tokens(self, messages, model="gpt-4o"):
        tokenizer = None

        # API 模式（如 qwen_dashscope）不应尝试从 HuggingFace 拉取 tokenizer
        # 仅当 llm_local_path 是本地存在路径时才用 transformers tokenizer
        if self.model_type != "qwen_dashscope" and self.llm_local_path and os.path.exists(self.llm_local_path):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.llm_local_path,
                    local_files_only=True,
                )
            except Exception:
                tokenizer = None

        if tokenizer is None:
            # 百炼 API 模式下回退到 tiktoken 估算
            tokenizer = tiktoken.encoding_for_model(model)
        
        full_message = [Message(**x) for x in messages]
        full_prompt = build_text_completion_prompt(full_message, allow_special=True)
        
        return len(tokenizer.encode(full_prompt))

    def _decomposer_call(self, msgs: list, model: str, max_tries: int = 3,
                         enable_thinking: bool = False, thinking_budget: int = 8192) -> str:
        """专用于分解器的 LLM 调用：非流式，按需开启 thinking。"""
        api_key = os.getenv("DASHSCOPE_API_KEY", "EMPTY")
        api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        client = OpenAI(api_key=api_key, base_url=api_base)

        extra = {"enable_thinking": enable_thinking}
        if enable_thinking:
            extra["thinking_budget"] = thinking_budget

        for attempt in range(max_tries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    extra_body=extra,
                    temperature=0.6,
                )
                msg = resp.choices[0].message
                content = msg.content or ""
                if enable_thinking:
                    thinking = getattr(msg, "reasoning_content", None) or ""
                    if thinking:
                        est_tokens = len(thinking) // 4
                        print(f"[decomposer][{model}] thinking {len(thinking)} chars (~{est_tokens} tokens, budget={thinking_budget}) (discarded)")
                if content:
                    return content
            except Exception as e:
                print(f"[decomposer][{model}] attempt {attempt+1}/{max_tries} error: {e}")
                if attempt == max_tries - 1:
                    return ""
        return ""

    def decompose_question(self, question: str) -> str:
        """两步问题分解：先拆解，再校验，返回格式化的解题计划字符串。失败时返回空字符串。"""
        from prompt import DECOMPOSER_PROMPT, CHECKER_PROMPT

        decomposer_model = os.getenv("DECOMPOSER_MODEL", "qwen3.5-plus")
        checker_model = os.getenv("CHECKER_MODEL", "qwen3.5-flash")

        # ── Step 1: 分解（使用 DECOMPOSER_MODEL）─────────────
        decompose_msgs = [
            {"role": "user", "content": DECOMPOSER_PROMPT.format(question=question)}
        ]
        raw_plan = self._decomposer_call(decompose_msgs, model=decomposer_model, enable_thinking=True, thinking_budget=4096)
        if not raw_plan:
            return ""

        # 清理 markdown 代码块标记
        raw_plan = raw_plan.strip()
        if raw_plan.startswith("```"):
            raw_plan = raw_plan.split("\n", 1)[-1]
        if raw_plan.endswith("```"):
            raw_plan = raw_plan.rsplit("```", 1)[0]
        raw_plan = raw_plan.strip()

        # 提取 JSON 数组
        left = raw_plan.find('[')
        right = raw_plan.rfind(']')
        if left == -1 or right == -1:
            return ""
        raw_plan = raw_plan[left:right+1]

        try:
            steps = json.loads(raw_plan)
            if not isinstance(steps, list) or len(steps) == 0:
                return ""
        except Exception:
            return ""

        # 单步问题不走 checker
        if len(steps) > 1:
            # ── Step 2: 校验 / 精炼（使用 CHECKER_MODEL）────────
            checker_msgs = [
                {"role": "user", "content": CHECKER_PROMPT.format(
                    question=question,
                    plan=json.dumps(steps, ensure_ascii=False, indent=2)
                )}
            ]
            checked = self._decomposer_call(checker_msgs, model=checker_model, enable_thinking=True, thinking_budget=2048)
            if checked:
                checked = checked.strip()
                if checked.startswith("```"):
                    checked = checked.split("\n", 1)[-1]
                if checked.endswith("```"):
                    checked = checked.rsplit("```", 1)[0]
                checked = checked.strip()
                cl = checked.find('[')
                cr = checked.rfind(']')
                if cl != -1 and cr != -1:
                    try:
                        refined = json.loads(checked[cl:cr+1])
                        if isinstance(refined, list) and len(refined) > 0:
                            steps = refined
                    except Exception:
                        pass  # 校验失败保留原始分解


        plan_json = json.dumps(steps, ensure_ascii=False, indent=2)
        plan_text = (
            "\n[Research Plan]\n"
            "Use this machine-readable plan directly:\n"
            f"{plan_json}\n"
        )
        print(f"[decomposer] Plan JSON ({len(steps)} steps):\n{plan_json}")
        return plan_text

    def update_scratchpad(self, question: str, plan_text: str, pending_results: list,
                           current_scratchpad: str) -> str:
        """每3次工具调用触发：用 SCRATCHPAD_MODEL 提炼已知事实，返回更新后的 scratchpad 字符串。"""
        from prompt import SCRATCHPAD_PROMPT

        scratchpad_model = os.getenv("SCRATCHPAD_MODEL", "qwen3.5-plus")
        results_text = "\n\n---\n\n".join(pending_results) if pending_results else "(none)"

        msgs = [{"role": "user", "content": SCRATCHPAD_PROMPT.format(
            question=question,
            plan=plan_text or "(no plan)",
            previous_scratchpad=current_scratchpad or "(none yet)",
            new_tool_results=results_text,
        )}]

        new_facts = self._decomposer_call(msgs, model=scratchpad_model, max_tries=2, enable_thinking=True, thinking_budget=4096)
        if new_facts and len(new_facts.strip()) > 10:
            new_scratchpad = new_facts.strip()
            print(f"[scratchpad] Updated ({len(new_scratchpad)} chars):\n{new_scratchpad}\n")
            return new_scratchpad
        return current_scratchpad

    def _run(self, data: str, model: str, user_prompt: str, **kwargs) -> List[List[Message]]:
        scratchpad = ""              # 搜索黑板（局部变量，避免线程竞争）
        tool_call_count = 0           # 工具调用总次数
        pending_tool_results = []     # 上次黑板更新后新增的工具结果
        try:
            question = data['item']['question']
        except: 
            raw_msg = data['item']['messages'][1]["content"] 
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 

        answer = data['item'].get('answer', '')
        # 前置问题分解：对复杂问题生成解题计划，注入到 prompt 头部
        plan_text = self.decompose_question(question)
        if plan_text:
            final_user_prompt = user_prompt + question + plan_text
        else:
            final_user_prompt = user_prompt + question
        messages = [{"role": "system", "content": self.system_message}, {"role": "user", "content": final_user_prompt}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        while num_llm_calls_available > 0:
            round += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages, model=model)
            print(f'Round {round}: {content}')
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            messages.append({"role": "assistant", "content": content.strip()})
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call_str = content.split('<tool_call>')[1].split('</tool_call>')[0]
                tool_name = ''
                tool_args = {}
                try:
                    tool_call = json.loads(tool_call_str)
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('arguments', {})
                    result = self._call_tool(tool_name, tool_args)
                except:
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                result = "<tool_response>\n" + result + "\n</tool_response>"
                messages.append({"role": "user", "content": result})

                # ── 搜索黑板：每 3 次工具调用更新一次已知事实 ──────────
                tool_call_count += 1
                # 前缀带调用序号，scratchpad 模型直接用序号标注，不猜 step
                pending_tool_results.append(
                    f"[Call #{tool_call_count}: {tool_name}]\n"
                    f"Query/Args: {json.dumps(tool_args, ensure_ascii=False)[:300]}\n"
                    f"Result: {result[:3000]}"
                )
                if tool_call_count % 3 == 0:
                    scratchpad = self.update_scratchpad(question, plan_text, pending_tool_results, scratchpad)
                    pending_tool_results = []
                    if scratchpad:
                        # 黑板注入 system message（语义正确：背景知识/指令层）
                        messages[0]["content"] = (
                            self.system_message
                            + "\n\n## 📋 Current Research Facts (Scratchpad)\n"
                            + scratchpad
                            + "\n"
                        )
                        # 同时在最新 tool_response 末尾追加简短 reminder，
                        # 缓解 recency bias（模型更关注末尾内容）
                        messages[-1]["content"] += (
                            "\n\n[Reminder] Key confirmed facts are in the system Scratchpad above. "
                            "Do NOT re-search what is already confirmed there."
                        )
                        # 压缩历史：保留 messages[0](system) + messages[1](原始问题)
                        # + 最近 6 条消息（3轮对话），其余已被黑板摘要，可丢弃
                        if len(messages) > 8:
                            messages = messages[:2] + messages[-6:]
                            print(f"[scratchpad] History compressed: kept first 2 + last 6 messages")
            if '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                break
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            max_tokens = MAX_TOKEN_LENGTH
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token count exceeds limit: {token_count} > {max_tokens}")
                
                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>\n\nRemember: the content inside <answer>...</answer> must be pure text only — a concise noun, name, or number with NO explanation or extra sentences. If the answer is a number, give the integer value only. Match the language of the question."
                content = self.call_server(messages, model=model)
                messages.append({"role": "assistant", "content": content.strip()})
                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    answer = prediction
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'
                result = {
                    "question": question,
                    "answer": answer,
                    "rollout_id": data['rollout_id'],
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            answer = prediction
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        result = {
            "question": question,
            "answer": answer,
            "rollout_id": data['rollout_id'],
            "messages": messages,
            "prediction": prediction,
            "termination": termination
        }
        return result

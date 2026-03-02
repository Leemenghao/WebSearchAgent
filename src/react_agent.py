import json
import os
import re
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
from prompt import ACTION_FORCE_MSG, ACTION_FORCE_THRESHOLD, DECOMPOSER_PROMPT, DECOMPOSER_REFINE_PROMPT


MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 40))
MAX_TOKEN_LENGTH = int(os.getenv('MAX_LENGTH', 31 * 1024 - 500))

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

    def call_server(self, msgs, max_tries=10):
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
                        model=self.model,
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
                    # ── 修复：工具调用被拆分到 thinking/response 两个流 ────────
                    # Qwen3 thinking 模式下偶发：<tool_call> 开头在 response_content，
                    # 但 JSON 体和 </tool_call> 落在 thinking_content 里。
                    # 检测并重建完整的 <tool_call>...</tool_call>。
                    if ('<tool_call>' in response_content
                            and '</tool_call>' not in response_content
                            and '</tool_call>' in thinking_content):
                        tc_end = thinking_content.rfind('</tool_call>')
                        tc_body = thinking_content[:tc_end].strip()
                        print('[call_server] salvaging split tool_call from thinking stream')
                        response_content = f'<tool_call>\n{tc_body}\n</tool_call>'
                        thinking_content = ''  # 已消费，无需保留
                    # 将思考过程包裹在 <think> 标签中，与回复内容合并
                    if thinking_content:
                        content = f"<think>{thinking_content}</think>\n{response_content}"
                    else:
                        content = response_content
                else:
                    # 普通非流式调用
                    chat_response = client.chat.completions.create(
                        model=self.model,
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
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)
        except Exception:
            # 百炼 API 模式下无本地模型路径，回退到 tiktoken 估算
            tokenizer = tiktoken.encoding_for_model(model)
        
        full_message = [Message(**x) for x in messages]
        full_prompt = build_text_completion_prompt(full_message, allow_special=True)
        
        return len(tokenizer.encode(full_prompt))

    def decompose_question(self, question: str) -> str:
        """
        两轮思考式问题拆解器。
        Round 1（enable_thinking）：生成初步子任务列表。
        Round 2（enable_thinking）：自我审查并精化，输出最终 JSON。
        返回可直接注入上下文的格式化计划字符串；失败时返回空字符串，不影响主流程。
        """
        api_key = os.getenv("DASHSCOPE_API_KEY", "EMPTY")
        api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        decomposer_model = os.getenv("DECOMPOSER_MODEL", self.model)
        client = OpenAI(api_key=api_key, base_url=api_base)

        def _call(msgs, temperature):
            try:
                resp = client.chat.completions.create(
                    model=decomposer_model,
                    messages=msgs,
                    extra_body={"enable_thinking": True},
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[decompose] API error: {e}")
                return None

        # ── Round 1：初步拆解 ────────────────────────────────────────────────
        msgs = [{"role": "user", "content": DECOMPOSER_PROMPT.format(question=question)}]
        plan_raw = _call(msgs, temperature=0.7)
        if not plan_raw:
            return ""
        print(f"[decompose] Round 1 raw:\n{plan_raw}")

        # ── Round 2：审查与精化 ──────────────────────────────────────────────
        msgs += [
            {"role": "assistant", "content": plan_raw},
            {"role": "user", "content": DECOMPOSER_REFINE_PROMPT.format(previous_plan=plan_raw)},
        ]
        plan_final = _call(msgs, temperature=0.3) or plan_raw
        print(f"[decompose] Round 2 final:\n{plan_final}")

        # ── 解析 JSON → 格式化可读计划 ───────────────────────────────────────
        try:
            json_match = re.search(r'\[.*?\]', plan_final, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\[.*?\]', plan_raw, re.DOTALL)
            steps = json.loads(json_match.group())
            lines = ["[Question Decomposition Plan — resolve in order before answering]"]
            for s in steps:
                dep = s.get("depends_on", [])
                dep_str = f"  [requires step(s) {dep}]" if dep else ""
                lines.append(f"  Step {s['step']}: {s['task']}{dep_str}")
            lines.append("")
            plan_str = "\n".join(lines)
            print(f"[decompose] Formatted plan:\n{plan_str}")
            return plan_str
        except Exception as e:
            print(f"[decompose] JSON parse failed: {e}")
            return ""

    def _run(self, data: str, model: str, user_prompt: str, **kwargs) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except: 
            raw_msg = data['item']['messages'][1]["content"] 
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 

        answer = data['item'].get('answer', '')
        self.user_prompt = user_prompt
        # 前置拆解：将多跳问题拆解为有序子任务，注入上下文帮助 Agent 规划行动
        decomp_plan = self.decompose_question(question)
        full_question = (decomp_plan + question) if decomp_plan else question
        self.user_prompt = self.user_prompt + full_question
        messages = [{"role": "system", "content": self.system_message}, {"role": "user", "content": self.user_prompt}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        idle_rounds = 0          # 连续无行动轮次计数（无 <tool_call> 且无 <answer>）
        while num_llm_calls_available > 0:
            round += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages)
            print(f'Round {round}: {content}')
            # 先剥离 <think>...</think>，再做后续所有判断：
            # 若先检查 <tool_response>，thinking 块内出现该字符串会导致误截断，
            # 把真正的 <tool_call> 一并切掉。
            content = re.sub(r'^<think>.*?</think>\n?', '', content, flags=re.DOTALL).strip()
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            messages.append({"role": "assistant", "content": content})
            has_tool_call = '<tool_call>' in content and '</tool_call>' in content
            has_answer = '<answer>' in content and '</answer>' in content
            if has_tool_call:
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                try:
                    tool_call = json.loads(tool_call)
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('arguments', {})
                    result = self._call_tool(tool_name, tool_args)
                except:
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                result = "<tool_response>\n" + result + "\n</tool_response>"
                messages.append({"role": "user", "content": result})
            if has_answer:
                termination = 'answer'
                break
            # ── 强制行动兜底：连续无行动则注入警告 ────────────────────────────
            if not has_tool_call and not has_answer:
                idle_rounds += 1
                print(f'[action-force] idle_rounds={idle_rounds}')
                if idle_rounds >= ACTION_FORCE_THRESHOLD:
                    print(f'[action-force] injecting force-action message at round {round}')
                    messages.append({"role": "user", "content": ACTION_FORCE_MSG})
                    idle_rounds = 0  # 注入后重置，给模型一次机会
            else:
                idle_rounds = 0  # 有行动则重置计数
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            max_tokens = MAX_TOKEN_LENGTH
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token count exceeds limit: {token_count} > {max_tokens}")
                
                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, provide what you consider the most likely answer in the following format:\n<answer>your answer</answer>\n\nRemember: the content inside <answer>...</answer> must be pure text only — a concise noun, name, or number with NO explanation or extra sentences. If the answer is a number, give the integer value only. Match the language of the question."
                content = self.call_server(messages)
                content = re.sub(r'^<think>.*?</think>\n?', '', content, flags=re.DOTALL).strip()
                messages.append({"role": "assistant", "content": content})
                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
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

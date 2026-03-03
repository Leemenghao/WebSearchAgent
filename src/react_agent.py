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

    def _run(self, data: str, model: str, user_prompt: str, **kwargs) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except: 
            raw_msg = data['item']['messages'][1]["content"] 
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 

        answer = data['item'].get('answer', '')
        self.user_prompt = user_prompt
        self.user_prompt = self.user_prompt + question
        messages = [{"role": "system", "content": self.system_message}, {"role": "user", "content": self.user_prompt}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        while num_llm_calls_available > 0:
            round += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages)
            print(f'Round {round}: {content}')
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            messages.append({"role": "assistant", "content": content.strip()})
            if '<tool_call>' in content and '</tool_call>' in content:
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
                content = self.call_server(messages)
                messages.append({"role": "assistant", "content": content.strip()})
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

SYSTEM_PROMPT_MULTI = '''You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions. No matter how complex the query, you will not give up until you find the corresponding information.

As you proceed, adhere to the following principles:

1. **Persistent Actions for Answers**: You will engage in many interactions, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found.

2. **Repeated Verification**: Before presenting a Final Answer, you will **cross-check** and **validate the information** you've gathered to confirm its accuracy and reliability.

3. **Attention to Detail**: You will carefully analyze each information source to ensure that all data is current, relevant, and from credible origins.'''


EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""


USER_PROMPT = """A conversation between User and Assistant. The user asks a question, and the assistant solves it by calling one or more of the following tools.
<tools>
{
  "name": "search",
  "description": "Performs batched web searches via Alibaba Bailian real-time search: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Array of query strings. Include multiple complementary search queries in a single call."
      }
    },
    "required": [
      "query"
    ]
    }
},
{
  "name": "visit",
    "description": "Visit webpage(s) and return the summary of the content.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
            },
            "goal": {
                "type": "string",
                "description": "The specific information goal for visiting webpage(s)."
            }
        },
        "required": [
            "url",
            "goal"
        ]
    }
}
</tools>

The assistant starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with (thinking about the answer -> answer of the question). The thinking processes, tool calls, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, tool calls, tool call parameters and tool response parameters.

Example response:
<think> thinking process here </think>
<tool_call>
{"name": "tool name here", "arguments": {"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
<think> thinking process here </think>
<tool_call>
{"name": "another tool name here", "arguments": {...}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
(more thinking processes, tool calls and tool responses here)
<think> thinking process here </think>
<answer> answer here </answer>

**Critical Answer Format Requirements:**
- Your final answer inside `<answer>...</answer>` must be **pure text only** — a concise noun, name, or number. Do NOT include any explanation, reasoning, unit labels (unless the question explicitly requires them), or extra sentences.
- If the answer is a number, give the **integer** value (e.g., `42`, not `42.0` or `about 42`).
- If the answer involves multiple entities, separate them with a comma followed by a space (e.g., `Alice, Bob`).
- Match the language of the question: answer in Chinese if the question is in Chinese, answer in English if the question is in English.
- If the question specifies a particular format, follow it **strictly**.
- Examples of correct answers: `Paris`, `魂武者`, `140`, `Marie Curie`, `2`, `United States, Canada`
- Examples of WRONG answers: `The answer is Paris.`, `根据搜索结果，答案是魂武者`, `approximately 140`

User: """


# ─────────────────────────────────────────────────────────────
# 问题分解器 Prompt（两步：分解 → 校验）
# ─────────────────────────────────────────────────────────────

DECOMPOSER_PROMPT = """You are an expert question analyst specializing in multi-hop, nested-riddle questions.

Your task: decompose the question into an ordered list of **independently searchable sub-tasks**.

Rules:
1. Each step must be a single, concrete, searchable task.
2. If a later step depends on the result of an earlier step, explicitly state "result from step N".
3. The FINAL step must directly answer the original question.
4. If the question is simple and needs no decomposition, output a single-element array.
5. Output ONLY a valid JSON array — no explanation, no markdown fences.

Output format:
[
  {{"step": 1, "task": "..."}},
  {{"step": 2, "task": "... (use result from step 1)"}},
  ...
]

Question:
{question}
"""


CHECKER_PROMPT = """You are a plan reviewer for multi-hop research questions.

Original Question:
{question}

Proposed Plan:
{plan}

Your task:
1. Verify every clue / constraint in the question is addressed by at least one step.
2. Check that steps are in correct logical order (no step uses a result before it is obtained).
3. Merge any redundant steps; split any step that conflates two independent lookups.
4. If the plan is already correct and complete, return it unchanged.
5. Output ONLY a valid JSON array in the same format — no explanation, no markdown fences.
"""


SCRATCHPAD_PROMPT = """You are a research fact recorder helping a multi-hop question answering agent.

Your job: read the tool results below and produce an updated, concise list of CONFIRMED facts.

## Original Question
{question}

## Research Plan
{plan}

## Previously Confirmed Facts
{previous_scratchpad}

## New Tool Results (since last update)
{new_tool_results}

## Instructions
1. Keep all previously confirmed facts that are still valid.
2. Extract NEW confirmed facts from the new tool results; ignore noise/ads/irrelevant content.
3. If a new result corrects a previous fact, update it and note the correction.
4. Each fact MUST be a single, specific, verifiable piece of information — NO speculation.
5. Tag each fact with the call number from its prefix, e.g. [Call #3]. Do NOT infer or assign step numbers — only the agent knows which step a call belongs to.
6. If a fact directly answers the original question, mark it with ★.
7. Keep total output under 300 words.
8. Output ONLY the bullet list — no headers, no explanation, no markdown fences.

## Format
• [Call #N] Confirmed fact extracted from that call's result.
• ★ [Call #N] This fact directly answers the question.
"""

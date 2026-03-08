import asyncio
import json
import os
import re
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    # Reuse contest normalization from utils before streaming chunks.
    from utils.extract_submit import normalize as normalize_answer
except Exception:
    # Safe fallback if utils import is unavailable.
    def normalize_answer(text: str) -> str:
        return (text or "").strip().lower()


class InvokeRequest(BaseModel):
    question: str = Field(..., min_length=1)


app = FastAPI(title="WebSearchAgent LangStudio", version="1.0.0")
REQUEST_TIMEOUT_SECONDS = 570
PING_INTERVAL_SECONDS = 5.0


def _build_agent():
    try:
        from prompt import SYSTEM_PROMPT_MULTI, USER_PROMPT
        from react_agent import MultiTurnReactAgent
        import tool_search  # noqa: F401
        import tool_visit  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency: {exc.name}. Please ensure requirements.txt is installed."
        ) from exc

    model = os.getenv("DASHSCOPE_MAIN_MODEL", "qwen3-max")
    llm_cfg = {
        "model": model,
        "generate_cfg": {
            "max_input_tokens": 320000,
            "max_retries": 10,
            "temperature": 0.6,
            "top_p": 0.95,
        },
        "model_type": "qwen_dashscope",
    }
    system_message = SYSTEM_PROMPT_MULTI + "\nCurrent date: " + datetime.now().strftime("%Y-%m-%d")
    agent = MultiTurnReactAgent(
        llm=llm_cfg,
        function_list=["search", "visit"],
        system_message=system_message,
    )
    return agent, USER_PROMPT, model


def _run_once(req: InvokeRequest) -> Dict[str, Any]:
    question = req.question.strip()
    if not question:
        raise ValueError("question cannot be empty")

    agent, user_prompt, model = _build_agent()

    task = {
        "item": {"question": question, "answer": ""},
        "rollout_id": 1,
    }

    try:
        result = agent._run(task, model, user_prompt)
    except Exception as exc:
        raise RuntimeError(f"agent execution failed: {exc}") from exc

    return {
        "answer": str(result.get("answer", "")).strip(),
        "prediction": str(result.get("prediction", "")).strip(),
        "termination": str(result.get("termination", "")).strip(),
        "raw": result,
    }


def _to_answer_text(result: Dict[str, Any]) -> str:
    answer = result.get("answer", "") or ""
    prediction = result.get("prediction", "") or ""
    text = answer.strip() if answer.strip() else prediction.strip()
    if not text or text == "No answer found.":
        return "unknown"

    # Ensure pure text answer without markup wrappers.
    text = re.sub(r"</?answer>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    text = " ".join(text.splitlines()).strip()
    normalized = normalize_answer(text)
    return normalized if normalized else "unknown"


def _iter_text_chunks(text: str, chunk_size: int = 24):
    text = text or ""
    if not text:
        yield ""
        return
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


@app.post("/")
async def query(req: InvokeRequest, request: Request):
    accept = request.headers.get("accept", "")
    if "text/event-stream" not in accept:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_run_once, req),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return {
                "answer": _to_answer_text(result),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="request timeout") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def stream_response():
        try:
            task = asyncio.create_task(
                asyncio.wait_for(
                    asyncio.to_thread(_run_once, req),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            )
            while not task.done():
                yield "event: Ping\n\n"
                await asyncio.sleep(PING_INTERVAL_SECONDS)

            result = await task
            answer_text = _to_answer_text(result)
            for piece in _iter_text_chunks(answer_text):
                yield "event: Message\n"
                yield f"data: {json.dumps({'answer': piece}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
        except TimeoutError:
            yield "event: Message\n"
            yield f"data: {json.dumps({'answer': 'unknown'}, ensure_ascii=False)}\n\n"
        except Exception:
            yield "event: Message\n"
            yield f"data: {json.dumps({'answer': 'unknown'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")

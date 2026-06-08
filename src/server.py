"""
FastAPI server — wraps the gateway + inference stack.

Start:  uvicorn src.server:app --host 0.0.0.0 --port 8000
Or:     make run
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.gateway.rate_limiter import RateLimiter
from src.inference.engine import VLLMEngine, create_engine
from src.inference.worker import InferenceRequest, InferenceWorker
from src.streaming.sse import stream_sse

app = FastAPI(title="Fastino Labs Inference")

_rate_limiter = RateLimiter()
_workers: dict[str, InferenceWorker] = {}


def _get_worker(model_key: str) -> InferenceWorker:
    if model_key not in _workers:
        if model_key == "mock":
            engine = create_engine(use_mock=True)
        elif model_key.startswith("vllm:"):
            engine = VLLMEngine(model=model_key[5:])
        else:
            raise HTTPException(400, f"Unknown model key: {model_key!r}")
        _workers[model_key] = InferenceWorker(worker_id=model_key, engine=engine)
    return _workers[model_key]


def _build_prompt(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


@app.get("/api/models")
async def list_models() -> dict:
    return {
        "models": [
            {"id": "mock",                              "label": "Mock  (no inference — instant)"},
            {"id": "vllm:meta-llama/Llama-2-13b-chat-hf", "label": "Llama 2 13B  (vLLM, GPU required)"},
        ]
    }


class ChatRequest(BaseModel):
    messages: list[dict]
    model: str = "mock"
    max_tokens: int = 500
    tenant_id: str = "web_user"


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    worker = _get_worker(req.model)
    prompt = _build_prompt(req.messages)

    prompt_tokens = len(prompt.split())
    if not _rate_limiter.check_and_consume(req.tenant_id, prompt_tokens, req.max_tokens):
        raise HTTPException(429, "Rate limit exceeded — try again shortly")

    inference_req = InferenceRequest(
        prompt=prompt,
        max_tokens=req.max_tokens,
        tenant_id=req.tenant_id,
    )

    return StreamingResponse(
        stream_sse(worker, inference_req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

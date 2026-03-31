"""ASGI entrypoint for deploying the deep_research example on Render."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


EXAMPLE_DIR = Path(__file__).resolve().parent / "examples" / "deep_research"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

try:
    from agent import agent as research_agent
except Exception as exc:  # pragma: no cover - startup failure should surface clearly
    raise RuntimeError(
        "Failed to import the deep_research agent. Ensure dependencies are installed "
        "and the required environment variables are configured."
    ) from exc


app = FastAPI(
    title="DeepAgents Deep Research API",
    description="ASGI wrapper around the deep_research example for Render deployments.",
    version="1.0.0",
)


class InvokeRequest(BaseModel):
    """Request payload for invoking the research agent."""

    message: str = Field(..., min_length=1, description="User message to send to the agent.")


def _normalize_content(content: Any) -> str:
    """Convert LangChain message content into a simple response string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content)


def _run_agent(message: str) -> dict[str, Any]:
    """Invoke the deep research agent and normalize its response."""
    result = research_agent.invoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    messages = result.get("messages", [])
    if not messages:
        return {"output": "", "raw": result}
    final_message = messages[-1]
    return {
        "output": _normalize_content(getattr(final_message, "content", "")),
        "raw": result,
    }


@app.get("/")
async def root() -> dict[str, str]:
    """Basic service metadata endpoint."""
    return {
        "service": "deepagents-deep-research",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint for Render."""
    return {"status": "ok"}


@app.post("/invoke")
async def invoke(payload: InvokeRequest) -> dict[str, Any]:
    """Invoke the research agent with a user message."""
    try:
        return await asyncio.to_thread(_run_agent, payload.message)
    except Exception as exc:  # pragma: no cover - runtime errors are returned to clients
        raise HTTPException(status_code=500, detail=str(exc)) from exc

"""
NBA Insights API — FastAPI product surface for the multi-agent analyst.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from crewai_utils import (
    NBA_DATA_PATH,
    OPENAI_MODEL,
    clear_ask_cache,
    get_ask_cache_stats,
    run_ask,
)

app = FastAPI(
    title="NBA Insights API",
    description=(
        "Ask a natural-language basketball question. "
        "A CrewAI analyst uses structured data tools and returns an answer "
        "with citations, tool trace, cache status, and groundedness score."
    ),
    version="1.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language NBA question")
    csv_path: Optional[str] = Field(
        default=None,
        description="Optional path to NBA CSV (defaults to project nba24-25.csv)",
    )
    use_cache: bool = Field(default=True, description="Use in-memory TTL response cache")


class Citation(BaseModel):
    source: str
    tool: str
    snippet: Dict[str, Any]


class ToolTraceItem(BaseModel):
    tool: str
    ok: bool
    summary: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    tool_trace: List[ToolTraceItem]
    latency_ms: int
    model: str
    data_path: str
    cached: bool = False
    groundedness: Optional[float] = None
    refused: bool = False
    retries_used: int = 0


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": OPENAI_MODEL or os.getenv("OPENAI_MODEL", "gpt-4o"),
        "data_path": NBA_DATA_PATH,
        "data_exists": os.path.exists(NBA_DATA_PATH),
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "cache": get_ask_cache_stats(),
    }


@app.get("/cache/stats")
def cache_stats() -> Dict[str, Any]:
    return get_ask_cache_stats()


@app.post("/cache/clear")
def cache_clear() -> Dict[str, Any]:
    return clear_ask_cache()


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    try:
        result = run_ask(body.question, csv_path=body.csv_path, use_cache=body.use_cache)
        return AskResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ask failed: {e}") from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

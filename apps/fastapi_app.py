from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geoai_flood.env import load_project_env
from geoai_flood.geoai_tools import gee_live_status
from geoai_flood.query_pipeline import extract_coordinates, run_query_pipeline

ENV_PATH = load_project_env()


class AnalyzeRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=10,
        description="Natural-language business question containing one or more coordinates.",
        examples=[
            "Gudang logistik kami di Bekasi koordinat -6.23, 107.01 apakah layak dijadikan collateral kredit?"
        ],
    )
    mode: Literal["auto", "live", "local"] = Field(
        default="auto",
        description="FSI computation mode. Use live for GEE on-the-fly, local for raster fallback, auto for best available.",
    )
    provider: str | None = Field(
        default=None,
        description="Optional LLM provider override, for example groq or ollama.",
    )
    dry_run: bool = Field(
        default=False,
        description="If true, returns FSI and RAG context without LLM synthesis.",
    )
    config_path: str = Field(default="config/config.yaml")
    rag_config_path: str = Field(default="rag_config.json")


class CoordinatePreviewRequest(BaseModel):
    query: str = Field(..., min_length=1)


app = FastAPI(
    title="GeoAI Flood Risk Assistant API",
    description=(
        "FastAPI backend for first-pass flood-risk screening in Jabodetabek. "
        "It combines GeoAI/FSI point scoring, regulatory RAG retrieval, and optional LLM synthesis."
    ),
    version="1.0.0",
)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "GeoAI Flood Risk Assistant API",
        "status": "ready",
        "docs": "/docs",
        "health": "/health",
        "analyze": "/analyze",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    gee_status: dict[str, Any]
    try:
        gee_status = gee_live_status()
    except Exception as exc:
        gee_status = {"ready": False, "error": str(exc)}

    return {
        "status": "ok",
        "env_path": str(ENV_PATH) if ENV_PATH else None,
        "groq_api_key": "set" if bool(os.getenv("GROQ_API_KEY")) else "missing",
        "earthengine_project": os.getenv("EARTHENGINE_PROJECT") or None,
        "gee": gee_status,
    }


@app.post("/coordinates/preview")
def preview_coordinates(payload: CoordinatePreviewRequest) -> dict[str, Any]:
    points = extract_coordinates(payload.query)
    return {"query": payload.query, "count": len(points), "points": points}


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
    try:
        return run_query_pipeline(
            query=payload.query,
            config_path=payload.config_path,
            rag_config_path=payload.rag_config_path,
            mode=payload.mode,
            provider=payload.provider,
            dry_run=payload.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

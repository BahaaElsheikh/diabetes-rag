"""
FastAPI app for Day 1.

Endpoints:
    GET  /health                 - liveness check
    POST /search                 - raw retrieval (Layer 2), for the "show
                                    retrieved chunks" UI requirement
    POST /ask                    - retrieval + structured, cited answer
                                    (Layer 3 stub - swap the generator for
                                    an LLM call once Day 3 starts; the
                                    grounding contract stays the same)
    POST /patients/profile       - validate + echo back a patient profile
                                    (Day 1: manual entry, no persistence yet)

Run with:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.models.patient import PatientLabData
from src.retrieval.search import search

app = FastAPI(title="Diabetes RAG - Day 1")

# Local UI runs on a different port -> needs CORS during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class Citation(BaseModel):
    document_name: str
    section_number: str | None
    page_number: int


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[dict]
    refused: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search_endpoint(req: SearchRequest):
    results = search(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "results": [
            {
                "text": r.text,
                "citation": r.citation(),
                "document_name": r.document_name,
                "section_number": r.section_number,
                "page_number": r.page_number,
                "score": r.score,
            }
            for r in results
        ],
    }


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    """
    Day 1 stub for the Generation layer: no LLM call yet, just proves the
    grounding contract end to end -> if nothing relevant is retrieved, we
    refuse instead of guessing (Safety Layer, Day 4 will formalize this).
    """
    results = search(req.query, top_k=req.top_k)

    if not results:
        return AskResponse(
            answer=(
                "I don't have sufficient evidence in the indexed guideline "
                "to answer this. Please consult a healthcare professional."
            ),
            citations=[],
            retrieved_chunks=[],
            refused=True,
        )

    top = results[0]
    # Placeholder "generation": in Day 3 this becomes an LLM call whose
    # prompt is restricted to only these retrieved chunks.
    answer = f"Based on {top.document_name}: {top.text[:300]}..."

    return AskResponse(
        answer=answer,
        citations=[
            Citation(
                document_name=r.document_name,
                section_number=r.section_number,
                page_number=r.page_number,
            )
            for r in results
        ],
        retrieved_chunks=[{"text": r.text, "score": r.score} for r in results],
        refused=False,
    )


@app.post("/patients/profile")
def submit_patient_profile(profile: PatientLabData):
    # Day 1: validate only. Day 2+ would persist this and feed the Risk
    # Analysis layer.
    if profile.hba1c_percent is None and profile.fasting_glucose_mgdl is None:
        raise HTTPException(
            status_code=422,
            detail="Provide at least HbA1c or fasting glucose to build a profile.",
        )
    return {"received": True, "profile": profile}

"""
FastAPI app for Diabetes RAG (Day 3 - Grounded Generation & Citation).

Endpoints:
    GET  /health                 - Liveness check
    POST /search                 - Raw retrieval (Layer 2)
    POST /ask                    - Grounded LLM generation & programmatic citations (Layer 3)
    POST /patients/profile       - Patient profile validation
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.generation.llm_client import generate_response
from src.generation.prompts import STRICT_GROUNDING_SYSTEM_PROMPT, GroundedAnswer, build_user_prompt
from src.models.patient import PatientLabData
from src.retrieval.search import search
from src.ingestion.embedder import get_qdrant_client, ensure_collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diabetes_rag_api")
import asyncio

def _init_qdrant():
    try:
        client = get_qdrant_client()
        ensure_collection(client)
        logger.info("Qdrant collection successfully verified and ready.")
    except Exception as e:
        logger.warning(f"Could not connect or initialize Qdrant at startup: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Diabetes RAG API backend...")
    asyncio.create_task(asyncio.to_thread(_init_qdrant))
    yield
    logger.info("Shutting down Diabetes RAG API backend...")


app = FastAPI(title="Diabetes RAG - Day 3", lifespan=lifespan)

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


class ChunkDTO(BaseModel):
    text: str
    document_name: str
    section_number: str | None
    section_title: str | None
    page_number: int
    score: float
    patient_subgroup_tags: list[str] = []
    related_sections: list[str] = []


class LatencyBreakdown(BaseModel):
    retrieval_and_rerank_ms: float
    llm_ms: float
    total_ms: float


class AskResponse(BaseModel):
    recommendation: str
    supporting_excerpt: str
    citations: list[Citation]
    retrieved_chunks: list[ChunkDTO]
    refused: bool
    refusal_reason: str | None  # "no_relevant_chunks" | "llm_insufficient_evidence" | "unsupported_excerpt_hallucination" | None
    latency_ms: LatencyBreakdown
    answer: str | None = None  # Full formatted answer for UI displaytibility with frontend


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search_endpoint(req: SearchRequest):
    results = search(req.query, top_k=req.top_k, use_reranker=True)
    return {
        "query": req.query,
        "results": [
            {
                "text": r.text,
                "citation": r.citation(),
                "document_name": r.document_name,
                "section_number": r.section_number,
                "section_title": r.section_title,
                "page_number": r.page_number,
                "score": r.score,
            }
            for r in results
        ],
    }


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    """Day 3 Grounded Generation endpoint.

    Flow:
    1. Reranker retrieval pass. If 0 chunks pass reranker threshold, refuse immediately (Refusal Path 1).
    2. If chunks returned, format strict grounding prompt & call Gemini LLM with Pydantic JSON schema.
    3. If LLM returns INSUFFICIENT_EVIDENCE, refuse at LLM level (Refusal Path 2).
    4. Attach citations programmatically from retrieved chunk metadata.
    """
    t_start = time.perf_counter()

    # Step 1: Retrieval + Reranking pass
    t_retrieval_start = time.perf_counter()
    results = search(req.query, top_k=req.top_k, use_reranker=True)
    t_retrieval_end = time.perf_counter()
    retrieval_latency_ms = (t_retrieval_end - t_retrieval_start) * 1000

    chunk_dtos = [
        ChunkDTO(
            text=r.text,
            document_name=r.document_name,
            section_number=r.section_number,
            section_title=r.section_title,
            page_number=r.page_number,
            score=round(r.score, 4),
            patient_subgroup_tags=getattr(r, "patient_subgroup_tags", []),
            related_sections=getattr(r, "related_sections", []),
        )
        for r in results
    ]

    # --- Refusal Path 1: Reranker Level ---
    if not results:
        t_total_end = time.perf_counter()
        logger.info(f"[REFUSAL - Reranker] Query: '{req.query}' -> 0 relevant chunks retrieved. Skipping LLM call.")
        refusal_msg = (
            "I don't have sufficient evidence in the indexed guideline to answer this question. "
            "Please consult a healthcare professional."
        )
        return AskResponse(
            recommendation=refusal_msg,
            supporting_excerpt="",
            citations=[],
            retrieved_chunks=[],
            refused=True,
            refusal_reason="no_relevant_chunks",
            latency_ms=LatencyBreakdown(
                retrieval_and_rerank_ms=round(retrieval_latency_ms, 1),
                llm_ms=0.0,
                total_ms=round((t_total_end - t_start) * 1000, 1),
            ),
            answer=refusal_msg,
        )

    # Step 2: Format context and invoke LLM
    context_chunks = [
        {
            "section_number": r.section_number,
            "section_title": r.section_title,
            "page_number": r.page_number,
            "text": r.text,
            "patient_subgroup_tags": getattr(r, "patient_subgroup_tags", []),
        }
        for r in results
    ]
    user_prompt = build_user_prompt(req.query, context_chunks)

    t_llm_start = time.perf_counter()
    try:
        raw_llm_output = generate_response(
            system_prompt=STRICT_GROUNDING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=GroundedAnswer,
            temperature=0.0,
        )
        parsed_answer = GroundedAnswer.model_validate_json(raw_llm_output)
    except Exception as e:
        logger.error(f"LLM Generation or Parsing Error: {e}")
        # Fallback to safe refusal
        parsed_answer = GroundedAnswer(
            recommendation="INSUFFICIENT_EVIDENCE",
            supporting_excerpt="",
        )

    t_llm_end = time.perf_counter()
    llm_latency_ms = (t_llm_end - t_llm_start) * 1000

    rec = parsed_answer.recommendation.strip()
    excerpt = parsed_answer.supporting_excerpt.strip()

    # --- Refusal Path 2: LLM Level ---
    if rec.upper() == "INSUFFICIENT_EVIDENCE" or excerpt.upper() == "INSUFFICIENT_EVIDENCE":
        t_total_end = time.perf_counter()
        logger.info(f"[REFUSAL - LLM] Query: '{req.query}' -> LLM returned INSUFFICIENT_EVIDENCE.")
        refusal_msg = (
            "The retrieved guideline context does not contain sufficient specific evidence "
            "to answer this question accurately."
        )
        return AskResponse(
            recommendation=refusal_msg,
            supporting_excerpt="",
            citations=[],
            retrieved_chunks=chunk_dtos,
            refused=True,
            refusal_reason="llm_insufficient_evidence",
            latency_ms=LatencyBreakdown(
                retrieval_and_rerank_ms=round(retrieval_latency_ms, 1),
                llm_ms=round(llm_latency_ms, 1),
                total_ms=round((t_total_end - t_start) * 1000, 1),
            ),
            answer=refusal_msg,
        )

    # --- Refusal Path 3: Excerpt Validation Check ---
    if excerpt:
        excerpt_clean = " ".join(excerpt.lower().split())
        excerpt_found = False
        for r in results:
            chunk_text_clean = " ".join(r.text.lower().split())
            if (
                excerpt_clean in chunk_text_clean
                or excerpt_clean[:35] in chunk_text_clean
                or excerpt_clean[-35:] in chunk_text_clean
            ):
                excerpt_found = True
                break

        if not excerpt_found:
            t_total_end = time.perf_counter()
            logger.warning(
                f"[REFUSAL - UNGROUNDED EXCERPT] Query: '{req.query}' -> Excerpt validation failed. "
                f"Excerpt: {excerpt!r} not found in retrieved context."
            )
            refusal_msg = (
                "The supporting excerpt provided could not be verified in the retrieved guideline context."
            )
            return AskResponse(
                recommendation=refusal_msg,
                supporting_excerpt="",
                citations=[],
                retrieved_chunks=chunk_dtos,
                refused=True,
                refusal_reason="unsupported_excerpt_hallucination",
                latency_ms=LatencyBreakdown(
                    retrieval_and_rerank_ms=round(retrieval_latency_ms, 1),
                    llm_ms=round(llm_latency_ms, 1),
                    total_ms=round((t_total_end - t_start) * 1000, 1),
                ),
                answer=refusal_msg,
            )

    # Step 3: Programmatic Citation Attachment
    citations = [
        Citation(
            document_name=r.document_name,
            section_number=r.section_number,
            page_number=r.page_number,
        )
        for r in results
    ]

    t_total_end = time.perf_counter()
    total_ms = (t_total_end - t_start) * 1000
    logger.info(
        f"[SUCCESS] Query: '{req.query}' -> Answered cleanly. "
        f"Retrieval: {retrieval_latency_ms:.1f}ms, LLM: {llm_latency_ms:.1f}ms, Total: {total_ms:.1f}ms"
    )

    combined_answer = f"{rec}\n\nSupporting Excerpt: \"{excerpt}\""

    return AskResponse(
        recommendation=rec,
        supporting_excerpt=excerpt,
        citations=citations,
        retrieved_chunks=chunk_dtos,
        refused=False,
        refusal_reason=None,
        latency_ms=LatencyBreakdown(
            retrieval_and_rerank_ms=round(retrieval_latency_ms, 1),
            llm_ms=round(llm_latency_ms, 1),
            total_ms=round(total_ms, 1),
        ),
        answer=combined_answer,
    )


@app.post("/patients/profile")
def submit_patient_profile(profile: PatientLabData):
    if profile.hba1c_percent is None and profile.fasting_glucose_mgdl is None:
        raise HTTPException(
            status_code=422,
            detail="Provide at least HbA1c or fasting glucose to build a profile.",
        )
    return {"received": True, "profile": profile}

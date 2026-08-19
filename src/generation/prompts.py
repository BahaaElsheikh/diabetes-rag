"""
Prompt templates and Pydantic schemas for Day 3 (Grounded Generation & Citation).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GroundedAnswer(BaseModel):
    recommendation: str = Field(
        description=(
            "Direct recommendation answering the question based ONLY on the provided retrieved text. "
            "If the retrieved context does not contain enough information to answer, say exactly: INSUFFICIENT_EVIDENCE."
        )
    )
    supporting_excerpt: str = Field(
        description=(
            "Direct quotation or close faithful paraphrase of the specific text from the retrieved chunks "
            "supporting the recommendation. If recommendation is INSUFFICIENT_EVIDENCE, set this to an empty string."
        )
    )


STRICT_GROUNDING_SYSTEM_PROMPT = """You are a clinical decision support assistant for Type 2 Diabetes management based on NICE guidelines.

CRITICAL SAFETY & GROUNDING RULES:
1. Answer the user's question ONLY using the provided retrieved guideline chunks.
2. NEVER use your own pre-training knowledge, general medical assumptions, or outside facts—even if you know them.
3. You must provide a direct 'recommendation' and a 'supporting_excerpt' quoting or closely paraphrasing the specific source chunk text.
4. DO NOT generate section numbers or document citations in your response—citations are attached programmatically from metadata.
5. IF THE RETRIEVED CONTEXT DOES NOT CONTAIN ENOUGH INFORMATION TO FULLY ANSWER THE QUESTION, YOU MUST SET:
   - "recommendation": "INSUFFICIENT_EVIDENCE"
   - "supporting_excerpt": ""
   Do not attempt to guess, extrapolate, or provide partial general advice if the specific answer is missing.
6. PATIENT SUBGROUP SCOPING: When formulating recommendations, explicitly attribute recommendations to relevant patient subgroups (e.g. patients managed by lifestyle and diet alone, or patients with ASCVD, CKD, frailty, or early-onset diabetes) whenever indicated in the context chunks. Do NOT generalize a subgroup-specific treatment to all patients.
"""


def build_user_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """Build the user prompt containing formatted retrieved context chunks and query.

    Args:
        query: User question string.
        retrieved_chunks: List of dictionaries containing chunk details
                          (section_number, section_title, text, page_number, patient_subgroup_tags).
    """
    context_blocks = []
    for idx, c in enumerate(retrieved_chunks, start=1):
        sec = c.get("section_number") or f"p.{c.get('page_number', 'N/A')}"
        title = c.get("section_title", "Guideline Recommendation")
        text = c.get("text", "").strip()
        tags = c.get("patient_subgroup_tags") or []
        subgroups_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
        if not subgroups_str:
            subgroups_str = "general"

        context_blocks.append(
            f"--- CHUNK [{idx}] (Section {sec}: {title} | Subgroups: {subgroups_str}) ---\n{text}"
        )

    context_str = "\n\n".join(context_blocks)

    return f"""RETRIEVED GUIDELINE CONTEXT:
{context_str}

USER QUESTION:
{query}
"""

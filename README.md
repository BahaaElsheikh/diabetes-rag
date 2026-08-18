# Diabetes RAG - Day 1

Grounded RAG pipeline over **NICE NG28 (Type 2 Diabetes in Adults: Management)**.
Implements Layer 1 (Ingestion) and Layer 2 (Retrieval) fully, plus a stub
Layer 3 (Generation) that proves the grounding + refusal contract end to end.

## 1. Get the source PDF

Download the guideline PDF yourself (network policy on this machine blocks
`nice.org.uk`, so this step has to happen on your machine):

- https://www.nice.org.uk/guidance/ng28/resources/type-2-diabetes-in-adults-management-pdf-1837338615493

Save it as:

```
data/raw_pdfs/ng28.pdf
```

## 2. Run locally (fastest for development)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start Qdrant (needs Docker just for this one piece)
docker run -p 6333:6333 qdrant/qdrant:v1.11.0

# In another terminal: run the ingestion pipeline
python -m src.ingestion.run_ingestion data/raw_pdfs/ng28.pdf

# Sanity-check retrieval from the command line
python -m src.retrieval.search "HbA1c target for type 2 diabetes on lifestyle intervention alone"

# Start the API
uvicorn src.api.main:app --reload
```

Open `frontend/index.html` directly in your browser (double-click it, or
`python -m http.server 5500` inside `frontend/` and visit
`http://localhost:5500`). It talks to the API at `http://localhost:8000`.

## 3. Run everything with Docker Compose (closer to how you'd demo it)

```bash
docker compose up --build
```

This starts Qdrant + the FastAPI backend together. You still need to run
ingestion once (it's a one-off job, not a long-running service):

```bash
docker compose exec api python -m src.ingestion.run_ingestion data/raw_pdfs/ng28.pdf
```

Then open `frontend/index.html` in a browser - the API is on
`http://localhost:8000`.

## 4. What each layer does (maps to the hackathon spec)

| File | Layer | What it does |
|---|---|---|
| `src/ingestion/pdf_parser.py` | 1. Ingestion | Extracts text per page, strips NICE header/footer boilerplate |
| `src/ingestion/chunker.py` | 1. Ingestion | Splits on NICE's own numbered recommendations (e.g. `1.6.1`) instead of a fixed char count |
| `src/ingestion/embedder.py` | 1/2 | Embeds chunks (`bge-small-en-v1.5`) and upserts to Qdrant with citation metadata |
| `src/retrieval/search.py` | 2. Retrieval | Semantic search with a `score_threshold` - the refusal path when evidence is weak |
| `src/api/main.py` | 3. Generation (stub) + 4. Safety | `/ask` returns a structured, cited answer or refuses; `/search` exposes raw retrieved chunks for the "show your work" demo requirement |
| `src/models/patient.py` | Patient Health Profile | Structured schema for lab values (manual entry for Day 1; OCR is a later add-on, not required for the core score) |
| `frontend/index.html` | Day 5 demo | Minimal UI showing the answer, citations, and retrieved chunks side by side |

## 5. Known Day-1 simplifications (by design, not oversights)

- **Generation is a stub**, not an LLM call yet - `/ask` returns the top
  retrieved chunk directly. Wire in an LLM call inside `ask_endpoint` for
  Day 3, with a system prompt restricted to only the retrieved chunks.
- **OCR is deferred.** Lab values are entered manually via
  `POST /patients/profile`. Adding OCR later (e.g. `pytesseract` or a
  vision-LLM call) slots into the same `PatientLabData` schema without
  changing anything downstream.
- **`score_threshold=0.35`** in `search.py` is a starting point, not tuned -
  Day 2's job is to log real scores across test queries and adjust it.

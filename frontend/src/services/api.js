/**
 * API service for the Diabetes RAG backend.
 *
 * During development, Vite proxies /api/* → http://127.0.0.1:8000/*
 * so the browser never sends a cross-origin request and CORS is not an issue.
 *
 * In production builds you can set VITE_API_URL to an absolute origin
 * (e.g. https://your-railway-app.up.railway.app) and the requests will go
 * directly to that origin.  Make sure the FastAPI server has CORS enabled for
 * that origin in that case.
 */

const RAW_API_URL = import.meta.env.VITE_API_URL ?? '';

/**
 * Resolve the base URL:
 *   - In dev (proxy mode) RAW_API_URL is '' → we use relative paths (/api/...)
 *     which are rewritten by the Vite proxy.
 *   - When VITE_API_URL is set to an absolute URL we call that origin directly.
 */
function base() {
  if (!RAW_API_URL || RAW_API_URL === 'http://127.0.0.1:8000') {
    // Use the Vite dev-server proxy path in development.
    // In a production build served from the same origin this also works.
    return '/api';
  }
  return RAW_API_URL.replace(/\/$/, '');
}

/**
 * @typedef {Object} Citation
 * @property {string} document_name
 * @property {string|null} section_number
 * @property {number} page_number
 */

/**
 * @typedef {Object} ChunkDTO
 * @property {string} text
 * @property {string} document_name
 * @property {string|null} section_number
 * @property {string|null} section_title
 * @property {number} page_number
 * @property {number} score
 * @property {string[]} patient_subgroup_tags
 * @property {string[]} related_sections
 */

/**
 * @typedef {Object} LatencyBreakdown
 * @property {number} retrieval_and_rerank_ms
 * @property {number} llm_ms
 * @property {number} total_ms
 */

/**
 * @typedef {Object} AskResponse
 * @property {string} recommendation
 * @property {string} supporting_excerpt
 * @property {Citation[]} citations
 * @property {ChunkDTO[]} retrieved_chunks
 * @property {boolean} refused
 * @property {string|null} refusal_reason
 * @property {LatencyBreakdown} latency_ms
 * @property {string|null} answer
 */

/**
 * Call POST /ask on the FastAPI backend.
 *
 * @param {string} query   The user's clinical question.
 * @param {number} top_k   Number of chunks to retrieve (default 5).
 * @returns {Promise<AskResponse>}
 */
export async function askQuestion(query, top_k = 5) {
  const url = `${base()}/ask`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k }),
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => `HTTP ${res.status}`);
    throw new Error(`Backend error ${res.status}: ${detail}`);
  }

  return res.json();
}

/**
 * Call GET /health on the FastAPI backend.
 * Returns true when the backend is reachable.
 *
 * @returns {Promise<boolean>}
 */
export async function checkHealth() {
  try {
    const url = `${base()}/health`;
    const res = await fetch(url, { method: 'GET' });
    return res.ok;
  } catch {
    return false;
  }
}

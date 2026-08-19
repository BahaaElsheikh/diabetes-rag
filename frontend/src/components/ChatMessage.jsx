import CitationCard from './CitationCard';

/**
 * ChatMessage renders one exchange turn — either a user message or an
 * assistant response card.
 *
 * For assistant messages the full structured backend payload is rendered:
 *   - Status badge (Grounded Recommendation | Refused)
 *   - Recommendation text
 *   - Supporting excerpt (if present)
 *   - Citations
 *   - Refusal reason (if refused)
 *   - Latency breakdown
 */
export default function ChatMessage({ message }) {
  if (message.role === 'user') {
    return <UserMessage text={message.text} />;
  }

  if (message.role === 'error') {
    return <ErrorMessage text={message.text} />;
  }

  return <AssistantMessage data={message.data} />;
}

/* ── User bubble ─────────────────────────────────────────────── */
function UserMessage({ text }) {
  return (
    <div className="message-row user">
      <div className="avatar user" aria-hidden="true">You</div>
      <div className="message-body">
        <div className="bubble-user">{text}</div>
      </div>
    </div>
  );
}

/* ── Error card ──────────────────────────────────────────────── */
function ErrorMessage({ text }) {
  return (
    <div className="message-row assistant">
      <div className="avatar assistant" aria-hidden="true">🩺</div>
      <div className="message-body" style={{ maxWidth: '85%' }}>
        <div className="error-card" role="alert">
          <span className="error-icon">⚠️</span>
          <div>
            <strong>Backend unavailable</strong>
            <br />
            {text}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Assistant structured response ──────────────────────────── */
function AssistantMessage({ data }) {
  const {
    recommendation,
    supporting_excerpt,
    citations = [],
    refused,
    refusal_reason,
    latency_ms,
  } = data;

  return (
    <div className="message-row assistant">
      <div className="avatar assistant" aria-hidden="true">🩺</div>
      <div className="message-body" style={{ maxWidth: '85%' }}>
        <div className={`response-card ${refused ? 'refused' : ''}`}>

          {/* ── Status badge ── */}
          {refused ? (
            <span className="status-badge refused">⚠ Query Refused</span>
          ) : (
            <span className="status-badge grounded">✓ Grounded Recommendation</span>
          )}

          {/* ── Recommendation ── */}
          <p className="recommendation">{recommendation}</p>

          {/* ── Refusal reason ── */}
          {refused && refusal_reason && (
            <p style={{ fontSize: 12, color: 'var(--warning)', fontStyle: 'italic' }}>
              Refusal reason: <strong>{humaniseRefusalReason(refusal_reason)}</strong>
            </p>
          )}

          {/* ── Supporting excerpt ── */}
          {!refused && supporting_excerpt && (
            <div className="excerpt-section">
              <div className="excerpt-label">Supporting excerpt (NICE NG28)</div>
              <blockquote className="excerpt-box">"{supporting_excerpt}"</blockquote>
            </div>
          )}

          {/* ── Citations ── */}
          {!refused && citations.length > 0 && (
            <div className="citations-section">
              <div className="citations-label">
                📚 Citations ({citations.length})
              </div>
              <div className="citations-list">
                {citations.map((c, i) => (
                  <CitationCard key={i} citation={c} />
                ))}
              </div>
            </div>
          )}

          {/* ── Latency ── */}
          {latency_ms && (
            <div className="latency-row" aria-label="Response latency breakdown">
              <span className="latency-chip">
                🔍 Retrieval &amp; Rerank: <strong>{latency_ms.retrieval_and_rerank_ms} ms</strong>
              </span>
              <span className="latency-chip">
                🤖 LLM: <strong>{latency_ms.llm_ms} ms</strong>
              </span>
              <span className="latency-chip">
                ⏱ Total: <strong>{latency_ms.total_ms} ms</strong>
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* Map raw API refusal_reason codes to readable text. */
function humaniseRefusalReason(reason) {
  switch (reason) {
    case 'no_relevant_chunks':
      return 'No relevant guideline evidence found';
    case 'llm_insufficient_evidence':
      return 'LLM: insufficient evidence in retrieved context';
    case 'unsupported_excerpt_hallucination':
      return 'Generated excerpt could not be verified in source';
    default:
      return reason;
  }
}

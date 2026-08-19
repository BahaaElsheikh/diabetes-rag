/**
 * LoadingIndicator — animated three-dot typing indicator shown while
 * the backend is processing the /ask request.
 */
export default function LoadingIndicator() {
  return (
    <div className="loading-row">
      <div className="avatar assistant" aria-hidden="true">🩺</div>
      <div className="loading-card" role="status" aria-label="Retrieving evidence and generating recommendation…">
        <div className="dots" aria-hidden="true">
          <span className="dot" />
          <span className="dot" />
          <span className="dot" />
        </div>
        <span>Retrieving evidence &amp; generating grounded recommendation…</span>
      </div>
    </div>
  );
}

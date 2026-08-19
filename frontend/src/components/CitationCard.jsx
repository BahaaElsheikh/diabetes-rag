/**
 * CitationCard — renders a single citation returned by the backend.
 * Displays document name, section number, and page number.
 */
export default function CitationCard({ citation }) {
  const { document_name, section_number, page_number } = citation;

  const loc = section_number
    ? `Sec ${section_number} · p. ${page_number}`
    : `p. ${page_number}`;

  return (
    <span className="citation-card" title={`${document_name} — ${loc}`}>
      <span className="citation-icon">📑</span>
      <span className="citation-text">
        <span className="citation-doc">{document_name}</span>
        <span className="citation-loc">{loc}</span>
      </span>
    </span>
  );
}

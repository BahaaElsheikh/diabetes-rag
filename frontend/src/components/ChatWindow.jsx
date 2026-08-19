import { useEffect, useRef } from 'react';
import ChatMessage from './ChatMessage';
import LoadingIndicator from './LoadingIndicator';

const EXAMPLE_QUERIES = [
  'What is the recommended first-line drug treatment for Type 2 diabetes?',
  'What HbA1c target should be offered to a patient managed by lifestyle and diet alone?',
  'What treatment is recommended for adults with Type 2 diabetes and cardiovascular disease?',
  'What does NICE recommend for patients with Type 2 diabetes and chronic kidney disease?',
];

/**
 * ChatWindow — the scrollable message history area.
 *
 * Props:
 *   messages: Array  — all messages to display
 *   loading: boolean — whether a request is in flight
 *   onExampleClick(q: string) — fill input with an example query
 */
export default function ChatWindow({ messages, loading, onExampleClick }) {
  const bottomRef = useRef(null);

  /* Auto-scroll to bottom whenever messages/loading change */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  if (messages.length === 0 && !loading) {
    return (
      <div className="chat-area">
        <div className="empty-state">
          <div className="empty-icon" aria-hidden="true">🩺</div>
          <h2 className="empty-title">Diabetes Assistant</h2>
          <p className="empty-body">
            Ask clinical questions about Type 2 Diabetes management. All answers
            are grounded strictly in the <strong>NICE NG28</strong> guideline with
            supporting excerpts and citations.
          </p>
          <p style={{ fontSize: 12, color: 'var(--text-faint)', marginTop: -8 }}>
            Try one of these example queries:
          </p>
          <div className="example-queries">
            {EXAMPLE_QUERIES.map((q) => (
              <button
                key={q}
                className="example-btn"
                onClick={() => onExampleClick(q)}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-area" role="log" aria-live="polite" aria-label="Conversation">
      {messages.map((msg) => (
        <ChatMessage key={msg.id} message={msg} />
      ))}
      {loading && <LoadingIndicator />}
      <div ref={bottomRef} />
    </div>
  );
}

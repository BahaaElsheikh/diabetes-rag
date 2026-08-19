import { useCallback, useEffect, useRef, useState } from 'react';
import ChatWindow from './components/ChatWindow';
import ChatInput from './components/ChatInput';
import { askQuestion, checkHealth } from './services/api';
import './styles/index.css';

let nextId = 1;
const uid = () => String(nextId++);

export default function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading]   = useState(false);
  const [health, setHealth]     = useState('checking'); // 'checking' | 'online' | 'offline'
  const inputRef                = useRef(null);

  /* ── Health probe on mount ──────────────────────────────── */
  useEffect(() => {
    checkHealth().then((ok) => setHealth(ok ? 'online' : 'offline'));
  }, []);

  /* ── Send a query ───────────────────────────────────────── */
  const handleSend = useCallback(async (text) => {
    if (loading) return;

    // Append the user message immediately
    setMessages((prev) => [
      ...prev,
      { id: uid(), role: 'user', text },
    ]);
    setLoading(true);

    try {
      const data = await askQuestion(text, 5);

      setMessages((prev) => [
        ...prev,
        { id: uid(), role: 'assistant', data },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: 'error',
          text: `Could not reach the backend. Make sure FastAPI is running on port 8000.\n\nDetails: ${err.message}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, [loading]);

  /* ── Example query click (from empty state) ─────────────── */
  const handleExampleClick = useCallback((query) => {
    handleSend(query);
  }, [handleSend]);

  /* ── Clear conversation ─────────────────────────────────── */
  const handleClear = useCallback(() => {
    setMessages([]);
  }, []);

  const hasMessages = messages.length > 0;

  return (
    <div className="app-shell">
      {/* ─── Header ─────────────────────────────────────────── */}
      <header className="header">
        <div className="header-icon" aria-hidden="true">D</div>
        <div className="header-text">
          <div className="header-title">Diabetes Assistant</div>
          <div className="header-subtitle">
            Evidence-based Type 2 Diabetes guidance
            <span className="nice-badge">📋 NICE NG28</span>
            <span
              className={`health-dot ${health === 'online' ? 'online' : health === 'offline' ? 'offline' : ''}`}
              title={`Backend: ${health}`}
              aria-label={`Backend status: ${health}`}
            />
          </div>
        </div>
        {hasMessages && (
          <div className="header-actions">
            <button className="btn-clear" onClick={handleClear} aria-label="Clear conversation">
              Clear chat
            </button>
          </div>
        )}
      </header>

      {/* ─── Chat messages ───────────────────────────────────── */}
      <ChatWindow
        messages={messages}
        loading={loading}
        onExampleClick={handleExampleClick}
      />

      {/* ─── Input ───────────────────────────────────────────── */}
      <ChatInput onSend={handleSend} disabled={loading} ref={inputRef} />

      {/* ─── Disclaimer ──────────────────────────────────────── */}
      <p className="disclaimer">
        ⚕️ For clinical decision support only. Answers are grounded in NICE NG28 and must be interpreted by a qualified healthcare professional.
      </p>
    </div>
  );
}

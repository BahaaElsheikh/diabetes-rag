import { useCallback, useEffect, useRef, useState } from 'react';

const MAX_ROWS = 7; // max visible text rows before scrolling inside textarea

/**
 * ChatInput — sticky input bar at the bottom of the screen.
 *
 * Props:
 *   onSend(text: string) — called when the user submits a non-empty query
 *   disabled: boolean    — true while a request is in flight
 */
export default function ChatInput({ onSend, disabled }) {
  const [text, setText] = useState('');
  const textareaRef = useRef(null);

  /* Auto-grow the textarea up to MAX_ROWS lines */
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const lineH = parseInt(getComputedStyle(el).lineHeight, 10) || 22;
    const maxH   = lineH * MAX_ROWS;
    el.style.height = Math.min(el.scrollHeight, maxH) + 'px';
  }, [text]);

  const handleSubmit = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText('');
    // Reset height after clear
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  }, [text, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  const canSend = text.trim().length > 0 && !disabled;

  return (
    <div className="input-area">
      <div className="input-wrapper">
        <textarea
          ref={textareaRef}
          className="chat-textarea"
          placeholder="Ask a clinical question about Type 2 Diabetes management…"
          rows={1}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          aria-label="Clinical query input"
          aria-multiline="true"
        />
        <button
          className="send-btn"
          onClick={handleSubmit}
          disabled={!canSend}
          aria-label="Send query"
        >
          {/* Send arrow icon */}
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
          </svg>
        </button>
      </div>
      <p className="input-hint">
        Press <kbd>Enter</kbd> to send · <kbd>Shift+Enter</kbd> for a new line
      </p>
    </div>
  );
}

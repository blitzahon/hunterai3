import { useEffect, useRef, useState } from "react";
import Markdown from "./Markdown.jsx";

const SUGGESTIONS = [
  "Summarize my documents",
  "What topics do my documents cover?",
  "Explain the key ideas in simple terms",
];

function Message({ m }) {
  if (m.role === "user") {
    return (
      <div className="msg user">
        <div className="bubble">{m.content}</div>
      </div>
    );
  }
  return (
    <div className={`msg assistant ${m.error ? "failed" : ""}`}>
      <div className="avatar">H</div>
      <div className="bubble">{m.error ? <p>{m.content}</p> : <Markdown text={m.content} />}</div>
    </div>
  );
}

export default function Chat({ messages, pending, onSend, noIndex, onOpenDocs }) {
  const [text, setText] = useState("");
  const bottom = useRef(null);
  const box = useRef(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, pending]);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [text]);

  const submit = (value = text) => {
    const q = value.trim();
    if (!q || pending) return;
    setText("");
    onSend(q);
  };

  return (
    <section className="chat">
      <div className="scroll">
        <div className="thread">
          {messages.length === 0 && (
            <div className="empty">
              <div className="logo-mark big">H</div>
              <h2>What would you like to know?</h2>
              {noIndex ? (
                <p className="muted">
                  There are no indexed documents yet.{" "}
                  <button className="link" onClick={onOpenDocs}>
                    Upload some
                  </button>{" "}
                  to get started.
                </p>
              ) : (
                <div className="chips">
                  {SUGGESTIONS.map((s) => (
                    <button key={s} onClick={() => submit(s)}>
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          {messages.map((m, i) => (
            <Message key={i} m={m} />
          ))}
          {pending && (
            <div className="msg assistant">
              <div className="avatar">H</div>
              <div className="bubble typing">
                <span />
                <span />
                <span />
              </div>
            </div>
          )}
          <div ref={bottom} />
        </div>
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <textarea
          ref={box}
          rows={1}
          value={text}
          placeholder="Ask anything about your documents…"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <button type="submit" disabled={!text.trim() || pending} aria-label="Send">
          ↑
        </button>
      </form>
    </section>
  );
}

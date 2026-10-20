import { useEffect, useRef, useState } from "react";
import Markdown from "./Markdown.jsx";

const SUGGESTIONS = [
  { icon: "✨", title: "Summarize my documents", sub: "Get the big picture in seconds" },
  { icon: "🧭", title: "What topics do my documents cover?", sub: "Map out everything you've uploaded" },
  { icon: "💡", title: "Explain the key ideas simply", sub: "Plain-language breakdowns" },
  { icon: "🌐", title: "What's new on this topic?", sub: "Blend your docs with the live web" },
];

const fmt = (n) => (n || 0).toLocaleString();

// Small line under an answer: total tokens, with the breakdown on hover.
function Usage({ u }) {
  if (!u?.total_tokens) return null;
  const detail = `Input ${fmt(u.input_tokens)} · Output ${fmt(u.output_tokens)} · Thinking ${fmt(u.thinking_tokens)} · ${u.model_calls} model call${u.model_calls === 1 ? "" : "s"}`;
  return (
    <sub className="usage" title={detail}>
      ⚡ {fmt(u.total_tokens)} tokens
    </sub>
  );
}

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
      <div className="bubble">
        {m.error ? <p>{m.content}</p> : <Markdown text={m.content} />}
        <Usage u={m.usage} />
      </div>
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
              <h2>
                What would you like to <span className="grad-text">discover</span>?
              </h2>
              <p className="sub">Ask about your documents, or anything on the web.</p>
              {noIndex ? (
                <p className="muted">
                  There are no indexed documents yet.{" "}
                  <button className="link" onClick={onOpenDocs}>
                    Upload some
                  </button>{" "}
                  to get started.
                </p>
              ) : (
                <div className="cards">
                  {SUGGESTIONS.map((s) => (
                    <button key={s.title} onClick={() => submit(s.title)}>
                      <span className="icon">{s.icon}</span>
                      <strong>{s.title}</strong>
                      <small>{s.sub}</small>
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
        <div className="composer-box">
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
        </div>
        <button type="submit" disabled={!text.trim() || pending} aria-label="Send">
          ↑
        </button>
      </form>
    </section>
  );
}

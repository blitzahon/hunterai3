import { useEffect, useRef } from "react";

const GSI_SRC = "https://accounts.google.com/gsi/client";

export default function Login({ clientId, onCredential, error, busy }) {
  const buttonRef = useRef(null);
  const callbackRef = useRef(onCredential);
  callbackRef.current = onCredential;

  useEffect(() => {
    let cancelled = false;

    const init = () => {
      if (cancelled || !window.google?.accounts?.id || !buttonRef.current) return;
      window.google.accounts.id.initialize({
        client_id: clientId,
        callback: (res) => callbackRef.current(res.credential),
      });
      window.google.accounts.id.renderButton(buttonRef.current, {
        theme: "filled_black",
        size: "large",
        shape: "pill",
        text: "continue_with",
        width: 280,
      });
    };

    if (window.google?.accounts?.id) {
      init();
    } else {
      let script = document.querySelector(`script[src="${GSI_SRC}"]`);
      if (!script) {
        script = document.createElement("script");
        script.src = GSI_SRC;
        script.async = true;
        document.head.appendChild(script);
      }
      script.addEventListener("load", init);
    }
    return () => {
      cancelled = true;
    };
  }, [clientId]);

  return (
    <main className="login">
      <div className="login-card">
        <div className="logo-mark big">H</div>
        <h1 className="grad-text">Hunter AI</h1>
        <p>Ask questions about your documents. Sign in to continue.</p>
        <div className="features">
          <span>📄 Your documents</span>
          <span>🌐 Live web search</span>
          <span>⚡ Instant answers</span>
        </div>
        <div className="gsi" ref={buttonRef} />
        {busy && <p className="muted">Signing you in…</p>}
        {error && <p className="error">{error}</p>}
      </div>
    </main>
  );
}

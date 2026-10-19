import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import { useChats } from "./hooks.js";
import Login from "./components/Login.jsx";
import Chat from "./components/Chat.jsx";
import Documents from "./components/Documents.jsx";

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

export default function App() {
  const [boot, setBoot] = useState({ state: "loading" }); // loading | login | ready | error
  const [user, setUser] = useState(null);
  const [loginError, setLoginError] = useState("");
  const [loggingIn, setLoggingIn] = useState(false);

  // First load: is login required, and am I already signed in?
  useEffect(() => {
    (async () => {
      try {
        const cfg = await api.config();
        if (!cfg.loginRequired) return setBoot({ state: "ready", cfg });
        try {
          const me = await api.me();
          setUser(me.user);
          setBoot({ state: "ready", cfg });
        } catch (e) {
          if (e.status !== 401) throw e;
          setBoot({ state: "login", cfg });
        }
      } catch (e) {
        setBoot({ state: "error", message: e.message });
      }
    })();
  }, []);

  const onCredential = useCallback(async (credential) => {
    setLoggingIn(true);
    setLoginError("");
    try {
      const res = await api.login(credential);
      setUser(res.user);
      setBoot((b) => ({ ...b, state: "ready" }));
    } catch (e) {
      setLoginError(e.message);
    } finally {
      setLoggingIn(false);
    }
  }, []);

  const signOut = async () => {
    await api.logout().catch(() => {});
    window.google?.accounts?.id?.disableAutoSelect();
    setUser(null);
    setBoot((b) => ({ ...b, state: "login" }));
  };

  const expired = useCallback(() => {
    setUser(null);
    setBoot((b) => ({ ...b, state: "login" }));
  }, []);

  if (boot.state === "loading") return <div className="center muted">Loading…</div>;
  if (boot.state === "error") return <div className="center error">Could not reach the server: {boot.message}</div>;
  if (boot.state === "login")
    return <Login clientId={boot.cfg.googleClientId} onCredential={onCredential} error={loginError} busy={loggingIn} />;

  return <Workspace user={user} onSignOut={signOut} onExpired={expired} />;
}

function Workspace({ user, onSignOut, onExpired }) {
  const { chats, activeId, setActiveId, newChat, startChat, addMessage, removeChat } = useChats(user?.email || "local");
  const [pending, setPending] = useState(false);
  const [docs, setDocs] = useState([]);
  const [indexExists, setIndexExists] = useState(true);
  const [showDocs, setShowDocs] = useState(false);
  const [showNav, setShowNav] = useState(false);
  const [docStatus, setDocStatus] = useState("");
  const [toast, setToast] = useState("");
  const chatsRef = useRef(chats);
  chatsRef.current = chats;

  const flash = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(""), 4000);
  };

  const guard = useCallback(
    (e) => {
      if (e.status === 401) onExpired();
      return e.message;
    },
    [onExpired]
  );

  const refreshDocs = useCallback(async () => {
    try {
      const [d, h] = await Promise.all([api.documents(), api.health()]);
      setDocs(d.documents);
      setIndexExists(h.index_exists);
      return h;
    } catch (e) {
      guard(e);
    }
  }, [guard]);

  useEffect(() => {
    refreshDocs();
  }, [refreshDocs]);

  const active = chats.find((c) => c.id === activeId);
  const messages = active?.messages ?? [];

  const send = async (question) => {
    const userMsg = { role: "user", content: question };
    const history = messages.filter((m) => !m.error).slice(-16).map(({ role, content }) => ({ role, content }));
    const chatId = active ? active.id : startChat(userMsg);
    if (active) addMessage(chatId, userMsg);

    setPending(true);
    try {
      const res = await api.ask(question, history);
      addMessage(chatId, { role: "assistant", content: res.answer });
    } catch (e) {
      addMessage(chatId, { role: "assistant", error: true, content: guard(e) || "Something went wrong." });
    } finally {
      setPending(false);
    }
  };

  // After a save without inline indexing, wait for the bucket trigger to rebuild the index.
  const waitForIndex = async (before) => {
    for (let i = 0; i < 40; i++) {
      await wait(3000);
      const h = await api.health();
      if (h.index_updated_at && h.index_updated_at !== before) return true;
    }
    return false;
  };

  const upload = async (files) => {
    try {
      setDocStatus(`Uploading ${files.length} file${files.length > 1 ? "s" : ""}…`);
      const before = (await api.health()).index_updated_at;
      const res = await api.upload(files);
      if (res.status === "saved") {
        setDocStatus("Indexing…");
        const ok = await waitForIndex(before);
        flash(ok ? "Documents indexed." : "Uploaded. Indexing is still running — it will be ready shortly.");
      } else {
        flash("Documents uploaded and indexed.");
      }
    } catch (e) {
      flash(guard(e));
    } finally {
      setDocStatus("");
      refreshDocs();
    }
  };

  const remove = async (name) => {
    try {
      setDocStatus(`Deleting ${name}…`);
      await api.remove(name);
      flash(`${name} deleted.`);
    } catch (e) {
      flash(guard(e));
    } finally {
      setDocStatus("");
      refreshDocs();
    }
  };

  return (
    <div className="app">
      <div className={`sidebar ${showNav ? "open" : ""}`}>
        <div className="brand">
          <div className="logo-mark">H</div>
          <span>Hunter AI</span>
        </div>
        <button
          className="primary"
          onClick={() => {
            newChat();
            setShowNav(false);
          }}
        >
          + New chat
        </button>
        <nav className="history">
          {chats.map((c) => (
            <div key={c.id} className={`hist-item ${c.id === activeId ? "active" : ""}`}>
              <button
                onClick={() => {
                  setActiveId(c.id);
                  setShowNav(false);
                }}
                title={c.title}
              >
                {c.title || "New chat"}
              </button>
              <button className="icon-btn" aria-label="Delete chat" onClick={() => removeChat(c.id)}>
                ✕
              </button>
            </div>
          ))}
        </nav>
        <div className="side-footer">
          <button className="ghost" onClick={() => setShowDocs(true)}>
            📄 Documents <span className="count">{docs.length}</span>
          </button>
          {user && (
            <div className="user">
              {user.picture ? <img src={user.picture} alt="" referrerPolicy="no-referrer" /> : <div className="avatar">{user.name[0]}</div>}
              <div className="who">
                <strong>{user.name}</strong>
                <small>{user.email}</small>
              </div>
              <button className="icon-btn" onClick={onSignOut} title="Sign out" aria-label="Sign out">
                ⏻
              </button>
            </div>
          )}
        </div>
      </div>

      <main className="main">
        <header className="topbar">
          <button className="icon-btn menu" onClick={() => setShowNav((v) => !v)} aria-label="Menu">
            ☰
          </button>
          <span className="title">{active?.title || "New chat"}</span>
        </header>
        <Chat messages={messages} pending={pending} onSend={send} noIndex={!indexExists} onOpenDocs={() => setShowDocs(true)} />
      </main>

      {showDocs && (
        <Documents docs={docs} status={docStatus} onUpload={upload} onDelete={remove} onClose={() => setShowDocs(false)} />
      )}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

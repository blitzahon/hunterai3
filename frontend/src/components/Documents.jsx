import { useRef, useState } from "react";

const ALLOWED = [".pdf", ".txt", ".md"];
const MAX_MB = 25;

const prettySize = (n) => (n > 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);

export default function Documents({ docs, status, onUpload, onDelete, onClose }) {
  const input = useRef(null);
  const [drag, setDrag] = useState(false);
  const [problem, setProblem] = useState("");
  const busy = Boolean(status);

  const pick = (fileList) => {
    const files = Array.from(fileList || []);
    const bad = files.find((f) => !ALLOWED.some((ext) => f.name.toLowerCase().endsWith(ext)));
    if (bad) return setProblem(`${bad.name}: only PDF, TXT and MD files are supported.`);
    const big = files.find((f) => f.size > MAX_MB * 1048576);
    if (big) return setProblem(`${big.name} is larger than ${MAX_MB} MB.`);
    setProblem("");
    if (files.length) onUpload(files);
  };

  return (
    <div className="overlay" onClick={onClose}>
      <aside className="panel" onClick={(e) => e.stopPropagation()}>
        <header>
          <h2>Documents</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>

        <div
          className={`dropzone ${drag ? "drag" : ""} ${busy ? "disabled" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            if (!busy) pick(e.dataTransfer.files);
          }}
          onClick={() => !busy && input.current?.click()}
        >
          <input
            ref={input}
            type="file"
            multiple
            hidden
            accept={ALLOWED.join(",")}
            onChange={(e) => {
              pick(e.target.files);
              e.target.value = "";
            }}
          />
          {busy ? (
            <>
              <span className="spinner" />
              <strong>{status}</strong>
              <small>This can take a minute for large files.</small>
            </>
          ) : (
            <>
              <span className="drop-icon">📥</span>
              <strong>Drop files here or click to upload</strong>
              <small>PDF, TXT or MD · up to {MAX_MB} MB each</small>
            </>
          )}
        </div>
        {problem && <p className="error">{problem}</p>}

        <ul className="doclist">
          {docs.length === 0 && <li className="muted">No documents yet.</li>}
          {docs.map((d) => (
            <li key={d.name}>
              <span className={`doc-ext ${d.type.replace(".", "")}`}>{d.type.replace(".", "").toUpperCase()}</span>
              <span className="doc-name" title={d.name}>
                {d.name}
              </span>
              <span className="muted">{prettySize(d.size)}</span>
              <button
                className="icon-btn danger"
                disabled={busy}
                aria-label={`Delete ${d.name}`}
                onClick={() => window.confirm(`Delete ${d.name}?`) && onDelete(d.name)}
              >
                🗑
              </button>
            </li>
          ))}
        </ul>
      </aside>
    </div>
  );
}

async function request(path, options = {}) {
  const res = await fetch(path, { credentials: "same-origin", ...options });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not json */
    }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

const json = (method, body) => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  config: () => request("/api/config"),
  me: () => request("/api/auth/me"),
  login: (credential) => request("/api/auth/google", json("POST", { credential })),
  logout: () => request("/api/auth/logout", { method: "POST" }),
  health: () => request("/api/health"),
  ask: (question, history) => request("/api/ask", json("POST", { question, history })),
  documents: () => request("/api/documents"),
  upload: (files) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return request("/api/documents/upload", { method: "POST", body: form });
  },
  remove: (name) => request(`/api/documents/${encodeURIComponent(name)}`, { method: "DELETE" }),
};

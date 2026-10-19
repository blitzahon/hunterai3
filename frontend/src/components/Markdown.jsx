import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function CodeBlock({ children, className }) {
  const [copied, setCopied] = useState(false);
  const lang = /language-(\w+)/.exec(className || "")?.[1];
  const text = String(children).replace(/\n$/, "");

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked */
    }
  };

  return (
    <div className="codeblock">
      <div className="codeblock-bar">
        <span>{lang || "code"}</span>
        <button onClick={copy}>{copied ? "Copied" : "Copy"}</button>
      </div>
      <pre>
        <code>{text}</code>
      </pre>
    </div>
  );
}

const components = {
  // Block code arrives as <pre><code>; inline code stays a plain <code>.
  pre: ({ children }) => children,
  code: ({ className, children }) => {
    const isBlock = /language-/.test(className || "") || String(children).includes("\n");
    return isBlock ? <CodeBlock className={className}>{children}</CodeBlock> : <code className="inline">{children}</code>;
  },
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noreferrer noopener">
      {children}
    </a>
  ),
  table: ({ children }) => (
    <div className="table-wrap">
      <table>{children}</table>
    </div>
  ),
};

// Renders the model's Markdown: **bold**, *italic*, lists, tables, links, code blocks.
export default function Markdown({ text }) {
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  );
}

"""The agentic loop: the model decides when and how many times to retrieve
from the local knowledge base and/or search the web. Also holds short-term
conversation memory: prior turns are passed back in on each call so the
agent has context of what was already discussed.

Uses Groq (OpenAI-compatible tool-calling format).
"""

import json
import re

from ddgs import DDGS
from groq import BadRequestError, Groq

from rag.retriever import Retriever

SYSTEM_PROMPT = (
    "You answer questions using two tools: `retrieve` searches the local "
    "knowledge base of uploaded documents, and `search_web` searches the "
    "live internet for current or general information not in those "
    "documents. Use `retrieve` first for anything that might be in the "
    "user's documents. Use `search_web` for current events, general facts, "
    "or anything retrieval doesn't cover. You can call both if a question "
    "needs information from each.\n\n"
    "You have access to the conversation history. Use it to resolve "
    "references like 'it', 'that', or 'the one you mentioned' to what was "
    "discussed earlier, and don't ask the user to repeat context they "
    "already gave.\n\n"
    "Formatting rules for your final answer:\n"
    "- Be concise. Don't add section headers, horizontal rules, or extra "
    "blank lines unless the content genuinely needs that structure.\n"
    "- For code requests, give the code in a single code block with only a "
    "short intro line before it and a brief explanation after — skip "
    "restating obvious details.\n"
    "- Use clean, standard Markdown: '- ' for bullets, '**bold**' for "
    "emphasis, no stray asterisks or inconsistent spacing.\n"
    "- Use plain numbers with standard comma separators (e.g. 70,000), "
    "never spaced-out digit groups like '70 000'.\n"
    "- Do not copy raw phrasing or spacing artifacts from search results — "
    "rewrite everything in your own clean, tight prose.\n"
    "- Answer naturally without citing passage numbers or sources inline.\n\n"
    "If neither tool surfaces the answer, say so plainly instead of guessing."
)

RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Search the local knowledge base (the user's uploaded documents) "
            "for relevant passages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused search query"},
                "top_k": {
                    "type": "integer",
                    "description": "Number of passages to retrieve",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

SEARCH_WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the live internet for current information, general "
            "knowledge, or anything not covered by the local knowledge base."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The web search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

# How many prior turns (user+assistant pairs) to carry into each request.
# Keeps token usage bounded on long conversations instead of growing forever.
MAX_HISTORY_TURNS = 8


def search_web(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Web search failed: {e}"

    if not results:
        return "No web results found."

    formatted = []
    for r in results:
        title = re.sub(r"\s+", " ", r.get("title", "")).strip()
        body = re.sub(r"\s+", " ", r.get("body", "")).strip()
        href = r.get("href", "")
        formatted.append(f"[{title}]({href})\n{body}")
    return "\n\n".join(formatted)


class RAGAgent:
    def __init__(self, retriever: Retriever, model: str = "openai/gpt-oss-20b"):
        self.retriever = retriever
        self.model = model
        self.client = Groq()  # reads GROQ_API_KEY from env

    def answer(self, question: str, history: list[dict] | None = None, max_turns: int = 6) -> str:
        """Answer a question, optionally with prior conversation turns.

        `history` is a list of {"role": "user"|"assistant", "content": str}
        dicts representing the finished turns before this one — tool-call
        internals are not part of history, only the final text exchanged.
        """
        history = history or []
        # keep only the most recent turns so the request doesn't grow unbounded
        trimmed_history = history[-(MAX_HISTORY_TURNS * 2):]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *trimmed_history,
            {"role": "user", "content": question},
        ]

        retry_count = 0
        max_retries = 3

        for _ in range(max_turns):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=0,
                    tools=[RETRIEVE_TOOL, SEARCH_WEB_TOOL],
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    messages=messages,
                )
                retry_count = 0
            except BadRequestError as e:
                if "tool_use_failed" in str(e) and retry_count < max_retries:
                    retry_count += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your last tool call was malformed. Call the tool again "
                                "using exactly the correct JSON function-call format."
                            ),
                        }
                    )
                    continue
                return f"Model repeatedly failed to call the tool correctly: {e}"

            choice = response.choices[0]
            msg = choice.message

            if not msg.tool_calls:
                return msg.content or ""

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)

                if tc.function.name == "retrieve":
                    results = self.retriever.retrieve(args["query"], args.get("top_k", 5))
                    tool_output = "\n\n".join(
                        f"[{r['metadata']['source']} #{r['metadata']['chunk_index']}] {r['chunk']}"
                        for r in results
                    ) or "No relevant results found."

                elif tc.function.name == "search_web":
                    tool_output = search_web(args["query"], args.get("max_results", 5))

                else:
                    tool_output = f"Unknown tool: {tc.function.name}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_output,
                    }
                )

        return "Reached max reasoning turns without a final answer."
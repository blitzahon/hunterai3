"""The agentic loop: the model decides when and how many times to retrieve
from the local knowledge base and/or search the web. Also holds short-term
conversation memory: prior turns are passed back in on each call so the
agent has context of what was already discussed.

Uses Gemini on Vertex AI through the google-genai SDK. Authentication comes
from the Cloud Run service account (or `gcloud auth application-default
login` on your PC), so there is no API key.
"""

import os
import re

from ddgs import DDGS
from google import genai
from google.genai import errors, types

from rag.retriever import Retriever

SYSTEM_PROMPT = (
    "You answer questions using two tools: `retrieve` searches the local "
    "knowledge base of uploaded documents, and `search_web` searches the "
    "live internet for current or general information not in those "
    "documents. Use `retrieve` first for anything that might be in the "
    "user's documents. Use `search_web` for current events, general facts, "
    "or anything retrieval doesn't cover. You can call both if a question "
    "needs information from each. Call `search_web` at most twice per "
    "question, and if it reports that search is unavailable, don't call it "
    "again: answer with what you have and say what you couldn't verify.\n\n"
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

RETRIEVE_TOOL = types.FunctionDeclaration(
    name="retrieve",
    description=(
        "Search the local knowledge base (the user's uploaded documents) "
        "for relevant passages."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(
                type=types.Type.STRING, description="A focused search query"
            ),
            "top_k": types.Schema(
                type=types.Type.INTEGER,
                description="Number of passages to retrieve (default 5)",
            ),
        },
        required=["query"],
    ),
)

SEARCH_WEB_TOOL = types.FunctionDeclaration(
    name="search_web",
    description=(
        "Search the live internet for current information, general "
        "knowledge, or anything not covered by the local knowledge base."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(
                type=types.Type.STRING, description="The web search query"
            ),
            "max_results": types.Schema(
                type=types.Type.INTEGER,
                description="Number of results to return (default 5)",
            ),
        },
        required=["query"],
    ),
)

# How many prior turns (user+assistant pairs) to carry into each request.
# Keeps token usage bounded on long conversations instead of growing forever.
MAX_HISTORY_TURNS = 8

DEFAULT_MODEL = "gemini-2.5-flash"

# Each web search costs another model turn, and the scraped search backends often
# rate-limit cloud IPs, so cap searches per question instead of letting the model retry.
MAX_WEB_SEARCHES = int(os.getenv("MAX_WEB_SEARCHES", "2"))

SEARCH_UNAVAILABLE = (
    "Web search is unavailable right now. Do not call search_web again for this "
    "question. Answer from the documents and what you already know, and say "
    "plainly if you could not verify something."
)
SEARCH_LIMIT_REACHED = (
    "Web search limit for this question reached. Do not call search_web again. "
    "Answer from the information you already have."
)


class ModelBusyError(RuntimeError):
    """The model API is rate-limiting us (HTTP 429); the caller should try again later."""


class SearchUnavailableError(RuntimeError):
    pass


def search_web(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        raise SearchUnavailableError(str(e)) from e

    if not results:
        return "No web results found."

    formatted = []
    for r in results:
        title = re.sub(r"\s+", " ", r.get("title", "")).strip()
        body = re.sub(r"\s+", " ", r.get("body", "")).strip()
        href = r.get("href", "")
        formatted.append(f"[{title}]({href})\n{body}")
    return "\n\n".join(formatted)


def _history_to_contents(history: list) -> list:
    """Convert [{"role": "user"|"assistant", "content": str}] to Gemini Contents."""
    contents = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        text = (turn.get("content") or "").strip()
        if not text:
            continue
        role = "model" if turn.get("role") == "assistant" else "user"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=text)])
        )
    # A conversation must open with a user turn.
    while contents and contents[0].role == "model":
        contents.pop(0)
    return contents


class RAGAgent:
    def __init__(self, retriever: Retriever, model: str | None = None):
        self.retriever = retriever
        self.model = model or os.getenv("VERTEX_MODEL", DEFAULT_MODEL)

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is not set. Set it to your GCP project ID."
            )
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        # Uses Application Default Credentials: the Cloud Run service account
        # in the cloud, or `gcloud auth application-default login` locally.
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def _run_tool(self, name: str, args: dict, state: dict) -> str:
        try:
            if name == "retrieve":
                query = str(args.get("query", "")).strip()
                top_k = max(1, min(int(args.get("top_k") or 5), 10))
                results = self.retriever.retrieve(query, top_k)
                return (
                    "\n\n".join(
                        f"[{r['metadata']['source']} #{r['metadata']['chunk_index']}] {r['chunk']}"
                        for r in results
                    )
                    or "No relevant results found."
                )

            if name == "search_web":
                if state["search_blocked"]:
                    return SEARCH_UNAVAILABLE
                if state["searches"] >= MAX_WEB_SEARCHES:
                    return SEARCH_LIMIT_REACHED
                state["searches"] += 1
                query = str(args.get("query", "")).strip()
                max_results = max(1, min(int(args.get("max_results") or 5), 10))
                try:
                    return search_web(query, max_results)
                except SearchUnavailableError as e:
                    print(f"[Agent] Web search unavailable: {e}")
                    state["search_blocked"] = True
                    return SEARCH_UNAVAILABLE

            return f"Unknown tool: {name}"
        except Exception as e:
            return f"Tool {name} failed: {e}"

    def answer(self, question: str, history: list[dict] | None = None, max_turns: int = 6) -> str:
        """Answer a question, optionally with prior conversation turns.

        `history` is a list of {"role": "user"|"assistant", "content": str}
        dicts representing the finished turns before this one — tool-call
        internals are not part of history, only the final text exchanged.
        """
        history = history or []
        # keep only the most recent turns so the request doesn't grow unbounded
        trimmed_history = history[-(MAX_HISTORY_TURNS * 2):]

        contents = _history_to_contents(trimmed_history)
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=question)])
        )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            # Gemini 2.5+/3 "thinking" tokens count toward this limit, so keep it generous.
            max_output_tokens=4096,
            tools=[types.Tool(function_declarations=[RETRIEVE_TOOL, SEARCH_WEB_TOOL])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        # On the last turn tools are switched off, so the model has to write an answer
        # instead of spending every turn on more tool calls.
        final_config = config.model_copy(
            update={
                "tool_config": types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.NONE
                    )
                )
            }
        )

        malformed_retries = 0
        max_malformed_retries = 3
        tool_state = {"searches": 0, "search_blocked": False}

        for turn in range(max_turns):
            last_turn = turn == max_turns - 1
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=final_config if last_turn else config,
                )
            except errors.APIError as e:
                if e.code == 429:
                    raise ModelBusyError(str(e)) from e
                raise

            if not response.candidates:
                return "The model returned no answer (the request may have been blocked)."

            candidate = response.candidates[0]
            finish_reason = str(getattr(candidate, "finish_reason", "") or "")

            if "MALFORMED_FUNCTION_CALL" in finish_reason:
                if malformed_retries < max_malformed_retries:
                    malformed_retries += 1
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(
                                    text=(
                                        "Your last tool call was malformed. Call the tool "
                                        "again using exactly the correct function-call format."
                                    )
                                )
                            ],
                        )
                    )
                    continue
                return "Model repeatedly failed to call the tool correctly."

            function_calls = response.function_calls or []

            if not function_calls:
                return response.text or ""
            if last_turn:
                break

            # Keep the model's own turn verbatim (it can carry thought
            # signatures that Gemini needs to see again on the next call).
            contents.append(candidate.content)

            response_parts = []
            for fc in function_calls:
                tool_output = self._run_tool(fc.name, dict(fc.args or {}), tool_state)
                response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"output": tool_output},
                    )
                )
            if turn == max_turns - 2:
                response_parts.append(
                    types.Part.from_text(
                        text=(
                            "You have used all your tool calls. Write your final answer "
                            "now from the information you already have."
                        )
                    )
                )
            contents.append(types.Content(role="user", parts=response_parts))

        return "Sorry, I couldn't put together an answer this time. Please try rephrasing the question."
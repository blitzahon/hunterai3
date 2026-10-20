"""Agent loop tests with a fake Gemini client, so no network or GCP project is needed."""

from types import SimpleNamespace

import pytest
from google.genai import errors, types

from rag import agent as agent_mod
from rag.agent import ModelBusyError, RAGAgent


def _call(name, **args):
    return SimpleNamespace(name=name, args=args)


def _response(function_calls=None, text="", usage=None):
    candidate = SimpleNamespace(finish_reason="STOP", content=types.Content(role="model", parts=[]))
    return SimpleNamespace(candidates=[candidate], function_calls=function_calls, text=text, usage_metadata=usage)


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _agent(responses):
    a = RAGAgent.__new__(RAGAgent)
    a.retriever = SimpleNamespace(retrieve=lambda q, k: [])
    a.model = "fake"
    a.client = SimpleNamespace(models=FakeModels(responses))
    return a


def test_blocked_web_search_is_not_retried(monkeypatch):
    searches = []

    def blocked(query, max_results=5):
        searches.append(query)
        raise agent_mod.SearchUnavailableError("429 Too Many Requests")

    monkeypatch.setattr(agent_mod, "search_web", blocked)
    a = _agent([
        _response([_call("search_web", query="a")]),
        _response([_call("search_web", query="b")]),
        _response(text="done"),
    ])

    assert a.answer("q") == "done"
    assert searches == ["a"]


def test_web_searches_are_capped_per_question(monkeypatch):
    searches = []
    monkeypatch.setattr(agent_mod, "search_web", lambda q, max_results=5: searches.append(q) or "result")
    a = _agent([_response([_call("search_web", query=str(i))]) for i in range(4)] + [_response(text="done")])

    assert a.answer("q") == "done"
    assert len(searches) == agent_mod.MAX_WEB_SEARCHES


def test_model_rate_limit_raises_model_busy():
    a = _agent([errors.ClientError(429, {"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}})])
    with pytest.raises(ModelBusyError):
        a.answer("q")


class RecordingModels(FakeModels):
    def __init__(self, responses):
        super().__init__(responses)
        self.configs = []

    def generate_content(self, **kwargs):
        self.configs.append(kwargs["config"])
        return super().generate_content(**kwargs)


def test_last_turn_disables_tools_so_the_model_must_answer(monkeypatch):
    monkeypatch.setattr(agent_mod, "search_web", lambda q, max_results=5: "result")
    a = _agent([])
    models = RecordingModels([_response([_call("retrieve", query=str(i))]) for i in range(5)] + [_response(text="final answer")])
    a.client = SimpleNamespace(models=models)

    assert a.answer("q", max_turns=6) == "final answer"
    assert models.configs[-1].tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.NONE
    assert all(c.tool_config is None for c in models.configs[:-1])


def test_token_usage_is_summed_over_every_model_call(monkeypatch):
    def meta(p, c, t):
        return types.GenerateContentResponseUsageMetadata(
            prompt_token_count=p, candidates_token_count=c, thoughts_token_count=t, total_token_count=p + c + t
        )

    monkeypatch.setattr(agent_mod, "search_web", lambda q, max_results=5: "result")
    a = _agent([
        _response([_call("search_web", query="a")], usage=meta(100, 10, 50)),
        _response(text="done", usage=meta(300, 40, 0)),
    ])

    text, usage = a.answer_with_usage("q")
    assert text == "done"
    assert usage == {
        "input_tokens": 400, "output_tokens": 50, "thinking_tokens": 50, "total_tokens": 500, "model_calls": 2,
    }


class ContentsRecorder(FakeModels):
    def __init__(self, responses):
        super().__init__(responses)
        self.contents = []

    def generate_content(self, **kwargs):
        self.contents.append(list(kwargs["contents"]))
        return super().generate_content(**kwargs)


def _passage(text, source="notes.pdf", i=0):
    return {"chunk": text, "score": 0.8, "metadata": {"source": source, "chunk_index": i}}


def test_documents_are_searched_before_the_model_sees_the_question():
    queries = []
    a = _agent([])
    a.retriever = SimpleNamespace(retrieve=lambda q, k: queries.append(q) or [_passage("Sliding window keeps a range.")])
    models = ContentsRecorder([_response(text="done")])
    a.client = SimpleNamespace(models=models)

    assert a.answer("What is a sliding window?") == "done"
    assert queries == ["What is a sliding window?"]
    first_prompt = models.contents[0][-1].parts[0].text
    assert first_prompt.startswith("What is a sliding window?")
    assert "[notes.pdf #0] Sliding window keeps a range." in first_prompt


def test_web_search_is_refused_until_documents_are_checked(monkeypatch):
    searches = []
    monkeypatch.setattr(agent_mod, "search_web", lambda q, max_results=5: searches.append(q) or "web result")

    def flaky_retrieve(q, k, calls=[]):
        calls.append(q)
        if len(calls) == 1:
            raise RuntimeError("429 embedding quota")  # the automatic pre-search fails
        return [_passage("from docs")]

    a = _agent([
        _response([_call("search_web", query="early")]),
        _response([_call("retrieve", query="topic")]),
        _response([_call("search_web", query="later")]),
        _response(text="done"),
    ])
    a.retriever = SimpleNamespace(retrieve=flaky_retrieve)

    assert a.answer("q") == "done"
    assert searches == ["later"]

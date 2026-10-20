"""Agent loop tests with a fake Gemini client, so no network or GCP project is needed."""

from types import SimpleNamespace

import pytest
from google.genai import errors, types

from rag import agent as agent_mod
from rag.agent import ModelBusyError, RAGAgent


def _call(name, **args):
    return SimpleNamespace(name=name, args=args)


def _response(function_calls=None, text=""):
    candidate = SimpleNamespace(finish_reason="STOP", content=types.Content(role="model", parts=[]))
    return SimpleNamespace(candidates=[candidate], function_calls=function_calls, text=text)


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
    a.retriever = None
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

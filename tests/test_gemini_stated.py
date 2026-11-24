"""Unit tests for gemini_stated.py and GeminiClient, with fake clients (no requests to Google)."""

import json
from types import SimpleNamespace as NS

import gemini_stated as G
from gemini import GeminiClient, GeminiError


def test_parse_strict_lenient_and_unreadable():
    assert G.parse("Answer: yes\nProbability: 70") == ("yes", 70, True)
    assert G.parse("Some thought.\n**Answer:** No\n**Probability:** 0\n") == ("no", 0, False)
    assert G.parse("I cannot tell.") == (None, None, False)


class FakeModels:
    def __init__(self, response):
        self.response, self.calls = response, []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config.temperature, config.max_output_tokens, config.thinking_config and config.thinking_config.thinking_budget))
        return self.response


def test_gemini_client_reads_answer_text_and_tokens():
    parts = [NS(text="thinking...", thought=True), NS(text="Answer: no\nProbability: 20", thought=None)]
    resp = NS(candidates=[NS(content=NS(parts=parts), finish_reason=NS(name="STOP"))], usage_metadata=NS(prompt_token_count=900, candidates_token_count=8, thoughts_token_count=150))
    fake = FakeModels(resp)
    out = GeminiClient("gemini-x", client=NS(models=fake)).ask("question", 0, 100, 50)
    assert out == {"text": "Answer: no\nProbability: 20", "finish_reason": "STOP", "prompt_tokens": 900, "reply_tokens": 8, "reasoning_tokens": 150}
    assert fake.calls == [("gemini-x", "question", 0, 100, 50)]


class FakeClient:
    def __init__(self, model):
        self.model = model

    def ask(self, user, temperature, max_tokens, reasoning_budget=None):
        return {"text": "Answer: yes\nProbability: 80", "finish_reason": "STOP", "prompt_tokens": 1, "reply_tokens": 1, "reasoning_tokens": 0}


def test_run_caches_and_does_not_repeat(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(G, "OUT", tmp_path / "gemini.jsonl")
    assert G.main(["--models", "gemini-x", "--limit", "3"], client_for=FakeClient) == 0
    lines = [json.loads(s) for s in (tmp_path / "gemini.jsonl").read_text().splitlines()]
    assert [e["row"] for e in sorted(lines, key=lambda e: e["row"])] == [1, 2, 3]
    assert all(e["answer"] == "yes" and e["probability"] == 80 and e["strict"] for e in lines)
    assert G.main(["--models", "gemini-x", "--limit", "3"], client_for=FakeClient) == 0
    assert "every reply is cached" in capsys.readouterr().out
    assert len((tmp_path / "gemini.jsonl").read_text().splitlines()) == 3


def test_empty_reply_is_a_failure():
    import pytest
    resp = NS(candidates=[NS(content=NS(parts=None), finish_reason=NS(name="STOP"))], usage_metadata=NS(prompt_token_count=900, candidates_token_count=0, thoughts_token_count=None))
    with pytest.raises(GeminiError, match="empty reply"):
        GeminiClient("gemini-x", client=NS(models=FakeModels(resp))).ask("question", 0, 100)

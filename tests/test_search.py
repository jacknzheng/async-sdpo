"""Parallel Search client. No network -- requests.post is mocked."""

from unittest.mock import MagicMock

import pytest

from data.search import (
    WEB_SEARCH_TOOL,
    format_search_results,
    parallel_search,
)


def test_web_search_tool_schema_is_openai_shaped():
    fn = WEB_SEARCH_TOOL["function"]
    assert fn["name"] == "web_search"
    assert "search_queries" in fn["parameters"]["properties"]


def test_format_search_results_joins_title_url_excerpts():
    text = format_search_results(
        {
            "results": [
                {
                    "title": "Acme 10-K",
                    "url": "https://example.com/10k",
                    "excerpts": ["Revenue was $1B.", "Net income $50M."],
                },
                {"title": "Empty", "url": "", "excerpts": []},
            ]
        }
    )
    assert "Acme 10-K" in text
    assert "https://example.com/10k" in text
    assert "Revenue was $1B." in text
    assert "Net income $50M." in text


def test_format_search_results_empty():
    assert format_search_results({"results": []}) == "(no search results)"


def test_parallel_search_requires_api_key(monkeypatch):
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PARALLEL_API_KEY"):
        parallel_search({"search_queries": ["acme revenue"]})


def test_parallel_search_fast_mode_payload(monkeypatch):
    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "search_id": "search_1",
            "session_id": "session_abc",
            "results": [
                {
                    "title": "Hit",
                    "url": "https://example.com",
                    "excerpts": ["excerpt one"],
                }
            ],
        }
        return resp

    monkeypatch.setattr("data.search.requests.post", fake_post)
    text, session_id = parallel_search(
        {"objective": "Find Acme revenue", "search_queries": ["Acme revenue 2024"]},
        mode="fast",
        client_model="Qwen/Qwen3.8-27B",
    )
    assert captured["json"]["mode"] == "fast"
    assert captured["json"]["search_queries"] == ["Acme revenue 2024"]
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["json"]["client_model"] == "Qwen/Qwen3.8-27B"
    assert "excerpt one" in text
    assert session_id == "session_abc"


def test_parallel_search_fills_queries_from_objective(monkeypatch):
    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"search_id": "s", "session_id": "sess", "results": []}
        return resp

    monkeypatch.setattr("data.search.requests.post", fake_post)
    parallel_search({"objective": "What is Acme's deposit beta?"})
    assert captured["json"]["search_queries"] == ["What is Acme's deposit beta?"]


def test_parallel_search_http_error_is_observation(monkeypatch):
    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")

    def fake_post(url, headers=None, json=None, timeout=None):
        resp = MagicMock()
        resp.status_code = 422
        resp.text = '{"error": "bad"}'
        return resp

    monkeypatch.setattr("data.search.requests.post", fake_post)
    text, _ = parallel_search({"search_queries": ["x"]})
    assert text.startswith("web_search error: HTTP 422")

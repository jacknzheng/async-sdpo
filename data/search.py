"""Parallel Search API client for diligence-bench TIR.

https://docs.parallel.ai/search/search-quickstart
POST https://api.parallel.ai/v1/search  header x-api-key, mode=fast.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.parallel.ai/v1/search"

WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the live web for filings, news, figures, and other facts needed "
            "to answer a financial diligence question. Prefer a clear objective plus "
            "2-3 short keyword queries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": (
                        "Self-contained natural-language goal for the search, with "
                        "enough context to understand intent."
                    ),
                },
                "search_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Concise keyword queries, 3-6 words each. Provide 2-3 for "
                        "best results. Required by the search API."
                    ),
                },
            },
            "required": ["search_queries"],
        },
    },
}


def format_search_results(payload: dict) -> str:
    """Flatten ranked results + excerpts into a tool observation."""
    lines: list[str] = []
    for result in payload.get("results") or []:
        title = (result.get("title") or "").strip()
        url = (result.get("url") or "").strip()
        header = " ".join(part for part in (title, f"({url})" if url else "") if part)
        if header:
            lines.append(header)
        for excerpt in result.get("excerpts") or []:
            text = str(excerpt).strip()
            if text:
                lines.append(text)
        if lines and lines[-1] != "":
            lines.append("")
    body = "\n".join(lines).strip()
    return body or "(no search results)"


def _queries_and_objective(arguments: dict) -> tuple[list[str], str]:
    raw = arguments.get("search_queries") or arguments.get("queries") or []
    if isinstance(raw, str):
        queries = [raw] if raw.strip() else []
    elif isinstance(raw, list):
        queries = [str(q).strip() for q in raw if str(q).strip()]
    else:
        queries = []
    objective = str(arguments.get("objective") or arguments.get("query") or "").strip()
    if not queries and objective:
        queries = [objective[:80]]
    if not objective and queries:
        objective = queries[0]
    return queries, objective


def parallel_search(
    arguments: dict,
    *,
    mode: str = "fast",
    timeout: float = 30.0,
    max_chars: int = 12000,
    session_id: str | None = None,
    client_model: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str | None]:
    """Call Parallel Search. Returns (formatted excerpts, session_id)."""
    key = api_key if api_key is not None else os.environ.get("PARALLEL_API_KEY")
    if not key:
        raise RuntimeError(
            "PARALLEL_API_KEY is not set; diligence TIR needs Parallel Search "
            "(https://docs.parallel.ai/search/search-quickstart)"
        )
    queries, objective = _queries_and_objective(arguments)
    if not queries:
        return "web_search error: need search_queries (or an objective)", session_id

    body: dict[str, Any] = {
        "objective": objective or queries[0],
        "search_queries": queries,
        "mode": mode,
        "max_chars_total": max_chars,
    }
    if session_id:
        body["session_id"] = session_id
    if client_model:
        body["client_model"] = client_model

    try:
        response = requests.post(
            SEARCH_URL,
            headers={"Content-Type": "application/json", "x-api-key": key},
            json=body,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        logger.warning("parallel search request failed: %s", exc)
        return f"web_search error: {exc}", session_id

    if response.status_code >= 400:
        snippet = (response.text or "")[:400]
        logger.warning("parallel search HTTP %s: %s", response.status_code, snippet)
        return f"web_search error: HTTP {response.status_code}: {snippet}", session_id

    try:
        payload = response.json()
    except ValueError:
        return "web_search error: response was not JSON", session_id

    echoed = payload.get("session_id") or session_id
    text = format_search_results(payload)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n...(truncated)"
    return text, echoed

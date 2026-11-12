"""
devswarm.research.search_backend — Search backend adapters (Fake, SearXNG, Brave).

Provides a unified interface for web search across local and cloud backends.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Normalized search result item."""
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class SearchBackendError(Exception):
    """Base exception for search backend failures."""


class SearchBackend(ABC):
    """Abstract base class for search backends."""

    @abstractmethod
    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Execute web search and return top k results."""
        ...


class FakeSearchBackend(SearchBackend):
    """Deterministic in-memory mock search backend for tests."""

    def __init__(self, canned_results: dict[str, list[SearchResult]] | None = None) -> None:
        self.canned_results = canned_results or {}
        self.calls: list[tuple[str, int]] = []

    def add_canned_result(self, query: str, results: list[SearchResult]) -> None:
        self.canned_results[query.lower().strip()] = results

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        self.calls.append((query, k))
        q_norm = query.lower().strip()
        if q_norm in self.canned_results:
            return self.canned_results[q_norm][:k]

        # Default fallback results
        return [
            SearchResult(
                title=f"Result 1 for {query}",
                url=f"https://example.com/search?q={query}",
                snippet=f"Relevant documentation and guidelines for {query}.",
            ),
            SearchResult(
                title=f"Result 2 for {query}",
                url=f"https://example.org/docs/{query}",
                snippet=f"Technical specification and examples regarding {query}.",
            ),
        ][:k]


class SearXNGSearchBackend(SearchBackend):
    """SearXNG search client for local, private, keyless web search."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        url = f"{self.base_url}/search"
        params = {"q": query, "format": "json", "categories": "general"}
        headers = {"Accept": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.get(url, params=params, headers=headers)
                if resp.status_code != 200:
                    raise SearchBackendError(f"SearXNG returned HTTP {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
        except Exception as exc:
            raise SearchBackendError(f"SearXNG request failed: {exc}") from exc

        raw_results = data.get("results", [])
        results: list[SearchResult] = []
        for item in raw_results[:k]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                )
            )
        return results


class BraveSearchBackend(SearchBackend):
    """Brave Search API client (key-based)."""

    def __init__(self, api_key: str | None = None, timeout_s: float = 10.0) -> None:
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY")
        self.timeout_s = timeout_s
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if not self.api_key:
            raise SearchBackendError("BRAVE_SEARCH_API_KEY is not configured in environment or .env.")

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {"q": query, "count": min(k, 20), "safesearch": "moderate"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.get(self.base_url, params=params, headers=headers)
                if resp.status_code != 200:
                    raise SearchBackendError(f"Brave Search API returned HTTP {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
        except Exception as exc:
            raise SearchBackendError(f"Brave Search request failed: {exc}") from exc

        web_results = data.get("web", {}).get("results", [])
        results: list[SearchResult] = []
        for item in web_results[:k]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                )
            )
        return results


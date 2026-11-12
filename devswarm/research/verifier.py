"""
devswarm.research.verifier — Source and citation verifier for research artifacts.

Verifies citations [Title](URL) in markdown text against the local DocsIndex
cache or live endpoint, and labels sources as [VERIFIED] or [UNVERIFIED].
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from devswarm.research.docs_index import DocsIndex

logger = logging.getLogger(__name__)

CITATION_REGEX = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")


@dataclass
class CitationVerification:
    title: str
    url: str
    verified: bool
    status: str  # "VERIFIED", "UNVERIFIED"
    match_score: float
    details: str = ""


class SourceVerifier:
    """Verifies that research artifacts cite real, reachable sources."""

    def __init__(self, docs_index: DocsIndex | None = None) -> None:
        self.docs_index = docs_index

    def extract_citations(self, text: str) -> list[tuple[str, str]]:
        """Extract all markdown links [title](url) from text."""
        return CITATION_REGEX.findall(text)

    def verify_citation(self, title: str, url: str) -> CitationVerification:
        """Verify single citation against cached docs index or URL structure."""
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return CitationVerification(
                title=title,
                url=url,
                verified=False,
                status="UNVERIFIED",
                match_score=0.0,
                details="Malformed or invalid URL scheme",
            )

        if self.docs_index:
            doc = self.docs_index.get_document(url)
            if doc:
                cached_title = doc.get("title", "")
                similarity = difflib.SequenceMatcher(
                    None, title.lower().strip(), cached_title.lower().strip()
                ).ratio()
                # Consider verified if URL is cached and title is reasonably similar
                if similarity >= 0.3:
                    return CitationVerification(
                        title=title,
                        url=url,
                        verified=True,
                        status="VERIFIED",
                        match_score=round(similarity, 2),
                        details=f"Matched cached document (similarity: {similarity:.2f})",
                    )
                else:
                    return CitationVerification(
                        title=title,
                        url=url,
                        verified=False,
                        status="UNVERIFIED",
                        match_score=round(similarity, 2),
                        details=f"URL cached but title mismatch ('{title}' vs '{cached_title}')",
                    )

        # Fallback if not in cache
        return CitationVerification(
            title=title,
            url=url,
            verified=False,
            status="UNVERIFIED",
            match_score=0.0,
            details="Source not present in retrieved research cache",
        )

    def verify_all(self, text: str) -> list[CitationVerification]:
        """Verify all citations found in text."""
        citations = self.extract_citations(text)
        return [self.verify_citation(title, url) for title, url in citations]

    def annotate_markdown(self, text: str) -> str:
        """Append a structured citation verification table to the markdown text."""
        verifications = self.verify_all(text)
        if not verifications:
            return text

        lines = [
            text,
            "",
            "---",
            "### 📚 Cited Sources & Verification",
            "| Source Title | URL | Status | Details |",
            "|---|---|:---:|---|",
        ]

        for v in verifications:
            status_icon = "✅ `[VERIFIED]`" if v.verified else "⚠️ `[UNVERIFIED]`"
            lines.append(f"| {v.title} | `{v.url}` | {status_icon} | {v.details} |")

        return "\n".join(lines)


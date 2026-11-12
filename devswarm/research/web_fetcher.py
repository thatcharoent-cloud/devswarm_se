"""
devswarm.research.web_fetcher — Host-side HTTP fetcher with strict SSRF guards.

Features:
  - Scheme validation (http/https only)
  - Domain allowlist & denylist validation
  - SSRF Guard: resolves IP and rejects loopback, private, link-local, and cloud metadata addresses
  - Content-Type filtering (html, text, json only; blocks binaries)
  - Redirect, timeout, and response size limits (256 KB)
  - HTML text extraction (strips script, style, nav, footer, iframe)
  - Encloses output in <untrusted_web_content> delimiters
"""

from __future__ import annotations

import fnmatch
from html.parser import HTMLParser
import ipaddress
import logging
import socket
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

from devswarm.research.docs_index import DocsIndex

logger = logging.getLogger(__name__)


class FetchError(Exception):
    """Base exception for web fetch failures."""


class SSRFViolationError(FetchError):
    """Raised when a URL targets a private, loopback, or metadata address."""


class PolicyDomainForbiddenError(SSRFViolationError):
    """Raised when a URL domain is forbidden by policy."""


class ContentTypeNotAllowedError(FetchError):
    """Raised when response Content-Type is not readable text or HTML."""


class HTMLTextExtractor(HTMLParser):
    """Clean HTML parser that strips executable and navigational boilerplate."""

    IGNORED_TAGS = {
        "script", "style", "noscript", "svg", "iframe", "object",
        "embed", "nav", "footer", "header", "aside"
    }

    def __init__(self) -> None:
        super().__init__()
        self.text_chunks: list[str] = []
        self.title: str = ""
        self._in_title = False
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in self.IGNORED_TAGS:
            self._ignore_depth += 1
        elif tag_lower == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in self.IGNORED_TAGS and self._ignore_depth > 0:
            self._ignore_depth -= 1
        elif tag_lower == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignore_depth > 0:
            return
        if self._in_title:
            self.title += data.strip()
        else:
            text = data.strip()
            if text:
                self.text_chunks.append(text)

    def get_text(self) -> str:
        return " ".join(self.text_chunks)


class WebFetcher:
    """Safe host-side web fetcher."""

    def __init__(
        self,
        docs_index: DocsIndex | None = None,
        max_bytes: int = 262144,  # 256 KB
        timeout_s: float = 10.0,
        max_redirects: int = 5,
        domain_denylist: list[str] | None = None,
        domain_allowlist: list[str] | None = None,
    ) -> None:
        self.docs_index = docs_index
        self.max_bytes = max_bytes
        self.timeout_s = timeout_s
        self.max_redirects = max_redirects
        self.domain_denylist = domain_denylist or [
            "localhost", "127.*", "169.254.*", "10.*", "192.168.*", "172.16.*",
            "172.17.*", "172.18.*", "172.19.*", "172.20.*", "172.21.*", "172.22.*",
            "172.23.*", "172.24.*", "172.25.*", "172.26.*", "172.27.*", "172.28.*",
            "172.29.*", "172.30.*", "172.31.*", "*.internal", "*.local", "*.onion"
        ]
        self.domain_allowlist = domain_allowlist or ["*"]

    def validate_url(self, url: str) -> None:
        """Validate URL scheme, domain allowlist/denylist, and perform SSRF check."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise FetchError(f"Unsupported URL scheme '{parsed.scheme}'. Only http and https are allowed.")

        hostname = parsed.hostname
        if not hostname:
            raise FetchError("Invalid URL: missing hostname.")

        # 1. Denylist check on hostname string
        for pat in self.domain_denylist:
            if fnmatch.fnmatch(hostname, pat):
                raise PolicyDomainForbiddenError(f"Target host '{hostname}' matches forbidden domain pattern '{pat}'.")

        # 2. Allowlist check on hostname string
        if "*" not in self.domain_allowlist:
            allowed = any(fnmatch.fnmatch(hostname, pat) for pat in self.domain_allowlist)
            if not allowed:
                raise PolicyDomainForbiddenError(f"Target host '{hostname}' is not in allowed domain list.")

        # 3. SSRF Check: resolve IP and verify public
        try:
            addr_info = socket.getaddrinfo(hostname, None)
        except socket.gaierror as exc:
            raise FetchError(f"DNS resolution failed for '{hostname}': {exc}") from exc

        for entry in addr_info:
            ip_str = entry[4][0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue

            if ip.is_loopback:
                raise SSRFViolationError(f"SSRF guard: IP address {ip_str} is a loopback address.")
            if ip.is_private:
                raise SSRFViolationError(f"SSRF guard: IP address {ip_str} is a private network address.")
            if ip.is_link_local:
                raise SSRFViolationError(f"SSRF guard: IP address {ip_str} is a link-local address.")
            if ip.is_reserved:
                raise SSRFViolationError(f"SSRF guard: IP address {ip_str} is a reserved address.")
            if ip.is_multicast:
                raise SSRFViolationError(f"SSRF guard: IP address {ip_str} is a multicast address.")
            # Explicit AWS/GCP metadata check
            if ip_str == "169.254.169.254":
                raise SSRFViolationError("SSRF guard: Access to cloud metadata endpoint (169.254.169.254) is blocked.")

    def sanitize_html(self, html_content: str) -> tuple[str, str]:
        """Convert HTML to clean readable text and extract title."""
        parser = HTMLTextExtractor()
        try:
            parser.feed(html_content)
            title = parser.title or "Untitled Web Page"
            text = parser.get_text()
            return title, text
        except Exception:
            return "Untitled Web Page", html_content

    async def fetch(self, url: str) -> str:
        """Fetch page content, sanitize, wrap in untrusted boundary, and cache."""
        self.validate_url(url)

        headers = {
            "User-Agent": "DevSwarm-ResearchBot/1.0 (+https://github.com/google/devswarm)",
            "Accept": "text/html,application/xhtml+xml,application/json,text/plain;q=0.9",
        }

        async with httpx.AsyncClient(
            timeout=self.timeout_s,
            max_redirects=self.max_redirects,
            follow_redirects=True,
        ) as client:
            try:
                resp = await client.get(url, headers=headers)
            except httpx.HTTPError as exc:
                raise FetchError(f"HTTP request to {url} failed: {exc}") from exc

        if resp.status_code != 200:
            raise FetchError(f"Web server returned status {resp.status_code} for {url}")

        # Check content type
        content_type = resp.headers.get("content-type", "").lower()
        if not any(t in content_type for t in ("text/html", "text/plain", "application/json", "text/")):
            raise ContentTypeNotAllowedError(f"Non-text Content-Type not allowed: '{content_type}'")

        # Cap response bytes
        content_bytes = resp.content[:self.max_bytes]
        text_content = content_bytes.decode("utf-8", errors="replace")

        # Clean text
        if "html" in content_type:
            title, clean_text = self.sanitize_html(text_content)
        else:
            title = url
            clean_text = text_content

        now_iso = datetime.now(timezone.utc).isoformat()

        # Cache in DocsIndex
        if self.docs_index:
            snippet = clean_text[:300]
            self.docs_index.insert_document(
                url=url,
                title=title,
                snippet=snippet,
                content=clean_text,
                retrieved_at=now_iso,
            )

        # Delimit as untrusted external content
        return (
            f'<untrusted_web_content url="{url}" title="{title}" retrieved="{now_iso}">\n'
            f'{clean_text}\n'
            f'</untrusted_web_content>'
        )

"""
devswarm.tools.fs_tools — Secure filesystem tools confined to the task worktree.

Security principles:
  1. All access is strictly confined to the task's worktree directory.
  2. Traversal (".."), absolute paths, and escaping symlinks are rejected.
  3. Protected paths (.git/, .swarm/, .env*, config/*.yaml, secrets) cannot be read or modified.
  4. Role write-scope (glob list) is enforced on all write operations.
  5. All output is UNTRUSTED DATA, truncated and delimited.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Any

from devswarm.platform.events import redact_secrets

logger = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_CHARS = 6000

PROTECTED_GLOBS = [
    ".git", ".git/**", ".swarm", ".swarm/**",
    ".env", ".env.*", "*.pem", "*.key", "*id_rsa*",
    "config/models.yaml", "config/roles.yaml", "config/gates.yaml", "config/policy.yaml",
]


class SecurityPathError(PermissionError):
    """Raised when a path violates security boundaries."""


def is_protected_path(rel_path: str) -> bool:
    """Check if relative path matches any protected path glob."""
    norm = rel_path.strip("/\\")
    for pattern in PROTECTED_GLOBS:
        if fnmatch.fnmatch(norm, pattern) or fnmatch.fnmatch(Path(norm).name, pattern):
            return True
        if norm.startswith(".git") or norm.startswith(".swarm") or norm.startswith(".env"):
            return True
    return False


def is_in_write_scope(rel_path: str, write_scope: list[str] | set[str] | None) -> bool:
    """Check if a path matches at least one write-scope glob."""
    if not write_scope:
        return False
    norm = rel_path.strip("/\\")
    for pattern in write_scope:
        if fnmatch.fnmatch(norm, pattern) or fnmatch.fnmatch(Path(norm).name, pattern):
            return True
    return False


def resolve_safe_path(
    path_str: str,
    worktree_root: Path,
    for_write: bool = False,
    write_scope: list[str] | None = None,
) -> tuple[Path, str]:
    """Resolve and validate a path within worktree_root.

    Returns:
      (resolved_absolute_path, relative_path_str)
    Raises:
      SecurityPathError on traversal, symlink escape, protected path, or scope violation.
    """
    wt_root = worktree_root.resolve()

    # Reject absolute paths, null bytes, URL encoded traversal, and explicit traversal attempts
    cleaned = path_str.strip()
    if "\0" in cleaned:
        raise SecurityPathError(f"Null bytes are strictly forbidden: '{path_str}'")

    cleaned_lower = cleaned.lower()
    if "%2e" in cleaned_lower or "%2f" in cleaned_lower or "%5c" in cleaned_lower:
        raise SecurityPathError(f"URL-encoded path traversal is strictly forbidden: '{path_str}'")

    if cleaned.startswith("/") or cleaned.startswith("\\"):
        raise SecurityPathError(f"Absolute paths are forbidden: '{path_str}'")

    raw_parts = Path(cleaned).parts
    if ".." in raw_parts:
        raise SecurityPathError(f"Path traversal ('..') is strictly forbidden: '{path_str}'")

    # Resolve target
    target = (wt_root / cleaned).resolve()

    # Verify target stays inside worktree root (prevents symlink escapes)
    try:
        rel = target.relative_to(wt_root)
        rel_str = str(rel)
    except ValueError:
        raise SecurityPathError(f"Path escapes task worktree: '{path_str}'")

    # Check protected paths
    if is_protected_path(rel_str):
        raise SecurityPathError(f"Access to protected path '{rel_str}' is denied by security policy")

    # If write operation, check write scope
    if for_write and write_scope is not None:
        if not is_in_write_scope(rel_str, write_scope):
            raise SecurityPathError(
                f"Write to '{rel_str}' denied: path is outside allowed write_scope {write_scope}"
            )

    return target, rel_str


class FileSystemTools:
    """Filesystem operations bounded by task worktree and role scope."""

    def __init__(
        self,
        worktree_root: Path | str,
        write_scope: list[str] | None = None,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        self.worktree_root = Path(worktree_root).resolve()
        self.write_scope = write_scope
        self.max_output_chars = max_output_chars

    def _wrap(self, content: str, status: str = "SUCCESS") -> str:
        clean = redact_secrets(content)
        if len(clean) > self.max_output_chars:
            clean = clean[:self.max_output_chars] + f"\n... [TRUNCATED {len(clean) - self.max_output_chars} chars]"
        return f"<UNTRUSTED_TOOL_DATA status=\"{status}\">\n{clean}\n</UNTRUSTED_TOOL_DATA>"

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read lines from a file in the task worktree."""
        try:
            target, rel_str = resolve_safe_path(path, self.worktree_root, for_write=False)
            if not target.exists() or not target.is_file():
                return self._wrap(f"Error: File '{path}' does not exist.", status="ERROR")

            with open(target, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            s_idx = max(1, start_line)
            e_idx = min(total_lines, end_line) if end_line is not None else total_lines

            output_lines = [f"File: {rel_str} (Lines {s_idx}-{e_idx} of {total_lines})"]
            for i in range(s_idx - 1, e_idx):
                output_lines.append(f"{i + 1:4d} | {lines[i].rstrip()}")

            return self._wrap("\n".join(output_lines))
        except SecurityPathError as exc:
            return self._wrap(f"Security violation: {exc}", status="DENIED")
        except Exception as exc:
            return self._wrap(f"Read error: {exc}", status="ERROR")

    def write_file(self, path: str, content: str) -> str:
        """Create or overwrite a file within role write scope."""
        try:
            target, rel_str = resolve_safe_path(
                path, self.worktree_root, for_write=True, write_scope=self.write_scope
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)

            return self._wrap(f"Successfully wrote {len(content)} characters to '{rel_str}'.")
        except SecurityPathError as exc:
            return self._wrap(f"Security violation: {exc}", status="DENIED")
        except Exception as exc:
            return self._wrap(f"Write error: {exc}", status="ERROR")

    def edit_file(self, path: str, old_str: str, new_str: str) -> str:
        """Replace exact unique string match in a file within role write scope."""
        try:
            target, rel_str = resolve_safe_path(
                path, self.worktree_root, for_write=True, write_scope=self.write_scope
            )
            if not target.exists() or not target.is_file():
                return self._wrap(f"Error: File '{path}' does not exist.", status="ERROR")

            with open(target, "r", encoding="utf-8") as f:
                content = f.read()

            count = content.count(old_str)
            if count == 0:
                return self._wrap(
                    f"Error: Target old_str was not found in '{rel_str}'. Edit rejected.",
                    status="ERROR",
                )
            if count > 1:
                return self._wrap(
                    f"Error: Target old_str matched {count} times in '{rel_str}'. Edit must be unambiguous.",
                    status="ERROR",
                )

            new_content = content.replace(old_str, new_str, 1)
            with open(target, "w", encoding="utf-8") as f:
                f.write(new_content)

            return self._wrap(f"Successfully replaced 1 occurrence in '{rel_str}'.")
        except SecurityPathError as exc:
            return self._wrap(f"Security violation: {exc}", status="DENIED")
        except Exception as exc:
            return self._wrap(f"Edit error: {exc}", status="ERROR")

    def list_dir(self, path: str = ".") -> str:
        """List files and subdirectories within task worktree."""
        try:
            target, rel_str = resolve_safe_path(path, self.worktree_root, for_write=False)
            if not target.exists() or not target.is_dir():
                return self._wrap(f"Error: Directory '{path}' does not exist.", status="ERROR")

            entries = []
            for item in sorted(target.iterdir()):
                item_rel = str(item.relative_to(self.worktree_root))
                if is_protected_path(item_rel):
                    continue
                type_flag = "[DIR]" if item.is_dir() else "[FILE]"
                entries.append(f"{type_flag} {item.name}")

            display_dir = rel_str if rel_str != "." else "/"
            header = f"Directory contents of '{display_dir}' ({len(entries)} items):\n"
            return self._wrap(header + "\n".join(entries))
        except SecurityPathError as exc:
            return self._wrap(f"Security violation: {exc}", status="DENIED")
        except Exception as exc:
            return self._wrap(f"List error: {exc}", status="ERROR")

    def search(self, pattern: str, path: str = ".") -> str:
        """Search text occurrences in files within task worktree."""
        try:
            target, _ = resolve_safe_path(path, self.worktree_root, for_write=False)
            if not target.exists():
                return self._wrap(f"Error: Path '{path}' does not exist.", status="ERROR")

            matches: list[str] = []
            file_candidates = [target] if target.is_file() else target.rglob("*")

            for fpath in file_candidates:
                if not fpath.is_file():
                    continue
                rel_str = str(fpath.relative_to(self.worktree_root))
                if is_protected_path(rel_str):
                    continue

                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        for lineno, line in enumerate(f, start=1):
                            if pattern in line:
                                matches.append(f"{rel_str}:{lineno} | {line.strip()[:120]}")
                                if len(matches) >= 50:
                                    break
                except Exception:
                    continue

                if len(matches) >= 50:
                    matches.append("... [Additional matches capped at 50]")
                    break

            if not matches:
                return self._wrap(f"No occurrences of '{pattern}' found.")
            return self._wrap(f"Found {len(matches)} matches for '{pattern}':\n" + "\n".join(matches))
        except SecurityPathError as exc:
            return self._wrap(f"Security violation: {exc}", status="DENIED")
        except Exception as exc:
            return self._wrap(f"Search error: {exc}", status="ERROR")


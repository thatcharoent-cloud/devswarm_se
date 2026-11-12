"""
devswarm.tools.repo_map — Compact repository symbol map bounded by token budgets.

Produces a lightweight directory outline + top-level function and class signatures
to give specialist agents architectural context without exhausting context windows.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from devswarm.tools.fs_tools import is_protected_path

logger = logging.getLogger(__name__)

DEFAULT_REPO_MAP_MAX_CHARS = 6000
IGNORE_DIRS = {".git", ".swarm", ".venv", "__pycache__", "node_modules", ".pytest_cache", "logs"}


def extract_python_signatures(filepath: Path) -> list[str]:
    """Parse top-level class and function signatures using AST."""
    signatures: list[str] = []
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                signatures.append(f"  class {node.name}")
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args = [a.arg for a in item.args.args]
                        signatures.append(f"    def {item.name}({', '.join(args)})")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                signatures.append(f"  {prefix} {node.name}({', '.join(args)})")
    except Exception:
        pass
    return signatures


def generate_repo_map(repo_path: Path | str, max_chars: int = DEFAULT_REPO_MAP_MAX_CHARS) -> str:
    """Generate a compact AST-aware file and symbol map sized to max_chars."""
    root = Path(repo_path).resolve()
    if not root.exists():
        return "<UNTRUSTED_REPO_MAP>\n[Empty Repository]\n</UNTRUSTED_REPO_MAP>"

    lines = ["<UNTRUSTED_REPO_MAP>", "# Repository Map (Architecture & Signatures)"]

    for path in sorted(root.rglob("*")):
        # Skip ignored directories
        if any(part in IGNORE_DIRS for part in path.parts):
            continue

        rel = str(path.relative_to(root))
        if is_protected_path(rel):
            continue

        if path.is_file():
            lines.append(f"\n{rel}")
            if path.suffix == ".py":
                sigs = extract_python_signatures(path)
                lines.extend(sigs)

        # Budget guard check
        current_len = sum(len(l) + 1 for l in lines)
        if current_len >= max_chars:
            lines.append("\n... [REPO MAP TRUNCATED TO FIT TOKEN BUDGET]")
            break

    lines.append("</UNTRUSTED_REPO_MAP>")
    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... [TRUNCATED]</UNTRUSTED_REPO_MAP>"
    return result


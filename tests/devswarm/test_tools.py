"""
tests/devswarm/test_tools.py — Unit tests for fs_tools: path traversal, symlinks, scopes, and protected paths.
"""

import os
import pytest
from pathlib import Path

from devswarm.tools.fs_tools import FileSystemTools, SecurityPathError, resolve_safe_path


def test_path_traversal_rejection(tmp_path):
    tools = FileSystemTools(worktree_root=tmp_path)

    # Relative traversal attempt
    res = tools.read_file("../../etc/passwd")
    assert "<UNTRUSTED_TOOL_DATA status=\"DENIED\">" in res
    assert "Path traversal" in res

    # Absolute path attempt
    res2 = tools.read_file("/etc/passwd")
    assert "<UNTRUSTED_TOOL_DATA status=\"DENIED\">" in res2
    assert "Absolute paths are forbidden" in res2


def test_symlink_escape_rejection(tmp_path):
    wt_dir = tmp_path / "worktree"
    wt_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret_file = outside_dir / "secret.txt"
    secret_file.write_text("HOST_SECRET_DATA", encoding="utf-8")

    # Create symlink inside worktree pointing outside
    link_path = wt_dir / "escaped_link"
    try:
        os.symlink(str(secret_file), str(link_path))
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported on this platform/filesystem")

    tools = FileSystemTools(worktree_root=wt_dir)
    res = tools.read_file("escaped_link")
    assert "<UNTRUSTED_TOOL_DATA status=\"DENIED\">" in res
    assert "escapes task worktree" in res


def test_protected_path_rejection(tmp_path):
    tools = FileSystemTools(worktree_root=tmp_path)

    for p in [".git", ".git/config", ".swarm/events.jsonl", ".env", ".env.local", "config/models.yaml"]:
        res_read = tools.read_file(p)
        assert "<UNTRUSTED_TOOL_DATA status=\"DENIED\">" in res_read
        assert "protected path" in res_read

        res_write = tools.write_file(p, "bad content")
        assert "<UNTRUSTED_TOOL_DATA status=\"DENIED\">" in res_write
        assert "protected path" in res_write


def test_write_scope_enforcement(tmp_path):
    # Only allow writing inside src/**
    tools = FileSystemTools(worktree_root=tmp_path, write_scope=["src/**", "tests/**"])

    # In scope: src/app.py -> should succeed
    res1 = tools.write_file("src/app.py", "def app(): pass")
    assert "<UNTRUSTED_TOOL_DATA status=\"SUCCESS\">" in res1
    assert (tmp_path / "src" / "app.py").exists()

    # Out of scope: setup.py or package.json -> MUST be rejected
    res2 = tools.write_file("setup.py", "malicious_code = True")
    assert "<UNTRUSTED_TOOL_DATA status=\"DENIED\">" in res2
    assert "outside allowed write_scope" in res2
    assert not (tmp_path / "setup.py").exists()


def test_read_file_line_ranges(tmp_path):
    tools = FileSystemTools(worktree_root=tmp_path)
    sample_file = tmp_path / "sample.py"
    sample_file.write_text("\n".join([f"line {i}" for i in range(1, 21)]), encoding="utf-8")

    res = tools.read_file("sample.py", start_line=5, end_line=8)
    assert "<UNTRUSTED_TOOL_DATA status=\"SUCCESS\">" in res
    assert "Lines 5-8" in res
    assert "line 5" in res
    assert "line 8" in res
    assert "line 4" not in res
    assert "line 9" not in res


def test_edit_file_exact_unique_replacement(tmp_path):
    tools = FileSystemTools(worktree_root=tmp_path, write_scope=["src/**"])
    src_file = tmp_path / "src" / "calc.py"
    src_file.parent.mkdir(parents=True, exist_ok=True)
    src_file.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

    # Unique replacement: fix bug
    res = tools.edit_file("src/calc.py", "return a - b", "return a + b")
    assert "<UNTRUSTED_TOOL_DATA status=\"SUCCESS\">" in res
    assert "return a + b" in src_file.read_text()

    # Ambiguous replacement (matches twice): should reject
    src_file.write_text("foo\nbar\nfoo\n")
    res_ambig = tools.edit_file("src/calc.py", "foo", "baz")
    assert "<UNTRUSTED_TOOL_DATA status=\"ERROR\">" in res_ambig
    assert "matched 2 times" in res_ambig

    # Missing target replacement: should reject
    res_miss = tools.edit_file("src/calc.py", "nonexistent_string", "baz")
    assert "<UNTRUSTED_TOOL_DATA status=\"ERROR\">" in res_miss
    assert "was not found" in res_miss


def test_list_dir_and_search(tmp_path):
    tools = FileSystemTools(worktree_root=tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "alpha.py").write_text("TARGET_STRING = 42\n")
    (tmp_path / "src" / "beta.py").write_text("OTHER_STRING = 100\n")

    list_res = tools.list_dir("src")
    assert "alpha.py" in list_res
    assert "beta.py" in list_res

    search_res = tools.search("TARGET_STRING")
    assert "alpha.py" in search_res
    assert "TARGET_STRING" in search_res


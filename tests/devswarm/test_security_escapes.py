"""
tests/devswarm/test_security_escapes.py — Rigorous security escape and boundary tests.

Tests:
  1. Path traversal attacks (relative, absolute, null-byte, URL-encoded).
  2. Escaping symlinks pointing outside the task worktree.
  3. Protected paths reading and writing (.git/, .env*, config/*.yaml, secrets).
  4. Role-based write-scope enforcement.
  5. Forbidden command executions (git, network clients, sudo, docker, shell chains).
  6. Static security scanner against hardcoded secrets and dangerous AST (eval, exec).
  7. High-risk path detection (auth, payments, migrations, CI workflows).
  8. Secret redaction in tool outputs and logs.
"""

import os
from pathlib import Path
import pytest

from devswarm.tools.fs_tools import (
    FileSystemTools,
    SecurityPathError,
    resolve_safe_path,
    is_protected_path,
    is_in_write_scope,
)
from devswarm.platform.sandbox import check_command_allowed, STRICT_DENYLIST
from devswarm.gates import BuiltinSecurityScanner
from devswarm.platform.events import redact_secrets


def test_path_traversal_escapes_rejected(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / "calc.py").write_text("print('hello')", encoding="utf-8")

    # Relative directory escapes
    with pytest.raises(SecurityPathError, match="strictly forbidden"):
        resolve_safe_path("../../etc/passwd", worktree)

    with pytest.raises(SecurityPathError, match="strictly forbidden"):
        resolve_safe_path("../other_folder/file.py", worktree)

    with pytest.raises(SecurityPathError, match="strictly forbidden"):
        resolve_safe_path("calc.py/../../secret.txt", worktree)

    # Absolute paths
    with pytest.raises(SecurityPathError, match="Absolute paths are forbidden"):
        resolve_safe_path("/etc/passwd", worktree)

    with pytest.raises(SecurityPathError, match="Absolute paths are forbidden"):
        resolve_safe_path("/var/log/system.log", worktree)

    # Null bytes
    with pytest.raises(SecurityPathError, match="Null bytes are strictly forbidden"):
        resolve_safe_path("calc.py\0.txt", worktree)

    # URL-encoded traversal
    with pytest.raises(SecurityPathError, match="URL-encoded path traversal"):
        resolve_safe_path("..%2f..%2fetc/passwd", worktree)

    with pytest.raises(SecurityPathError, match="URL-encoded path traversal"):
        resolve_safe_path("%2e%2e/secret.py", worktree)


def test_escaping_symlink_rejected(tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("TOP SECRET CONTENT", encoding="utf-8")

    worktree = tmp_path / "worktree"
    worktree.mkdir()

    # Create escaping symlink inside worktree pointing outside
    symlink_path = worktree / "link_to_outside"
    try:
        os.symlink(str(outside_dir), str(symlink_path), target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks not supported in current environment")

    # Attempt to resolve through escaping symlink
    with pytest.raises(SecurityPathError, match="escapes task worktree"):
        resolve_safe_path("link_to_outside/secret.txt", worktree)


def test_protected_paths_denied(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".git").mkdir()
    (worktree / ".git" / "config").write_text("git config", encoding="utf-8")
    (worktree / ".env").write_text("API_KEY=123", encoding="utf-8")
    (worktree / ".env.production").write_text("SECRET=abc", encoding="utf-8")
    (worktree / "id_rsa").write_text("private key", encoding="utf-8")

    config_dir = worktree / "config"
    config_dir.mkdir()
    (config_dir / "policy.yaml").write_text("policy: strict", encoding="utf-8")
    (config_dir / "models.yaml").write_text("models: test", encoding="utf-8")

    fs = FileSystemTools(worktree_root=worktree, write_scope=["*"])

    # Attempt to read protected paths
    read_git = fs.read_file(".git/config")
    assert "Security violation" in read_git or "DENIED" in read_git

    read_env = fs.read_file(".env")
    assert "Security violation" in read_env or "DENIED" in read_env

    read_rsa = fs.read_file("id_rsa")
    assert "Security violation" in read_rsa or "DENIED" in read_rsa

    read_policy = fs.read_file("config/policy.yaml")
    assert "Security violation" in read_policy or "DENIED" in read_policy

    # Attempt to write to protected paths
    write_git = fs.write_file(".git/config", "hacked")
    assert "Security violation" in write_git or "DENIED" in write_git
    assert (worktree / ".git" / "config").read_text(encoding="utf-8") == "git config"

    write_env = fs.write_file(".env", "hacked")
    assert "Security violation" in write_env or "DENIED" in write_env


def test_write_scope_enforcement(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / "calc.py").write_text("def add(): pass\n", encoding="utf-8")
    (worktree / "app.py").write_text("def run(): pass\n", encoding="utf-8")

    # Scope permits only calc.py
    fs = FileSystemTools(worktree_root=worktree, write_scope=["calc.py"])

    # Permitted write
    edit_calc = fs.edit_file("calc.py", "def add(): pass", "def add(): return 1")
    assert "Successfully replaced" in edit_calc

    # Out of scope write rejected
    edit_app = fs.edit_file("app.py", "def run(): pass", "def run(): return 2")
    assert "outside allowed write_scope" in edit_app
    assert "return 2" not in (worktree / "app.py").read_text(encoding="utf-8")

    # Out of scope creation rejected
    create_new = fs.write_file("new_module.py", "content = 1")
    assert "outside allowed write_scope" in create_new
    assert not (worktree / "new_module.py").exists()


def test_command_security_and_strict_denylist() -> None:
    backend_allowlist = ["python", "python3", "pytest", "ruff", "mypy"]

    # Allowed commands
    ok, _ = check_command_allowed("pytest tests/ -v", backend_allowlist)
    assert ok is True

    ok, _ = check_command_allowed("python -m py_compile calc.py", backend_allowlist)
    assert ok is True

    # Denied: command not in role allowlist
    ok, reason = check_command_allowed("npm install", backend_allowlist)
    assert ok is False
    assert "not in the allowlist" in reason

    # Denied: strictly forbidden commands (git, curl, wget, sudo, docker)
    for forbidden in ["git push", "git commit", "curl https://example.com", "wget http://...", "sudo ls", "docker run"]:
        ok, reason = check_command_allowed(forbidden, None)
        assert ok is False
        assert "forbidden" in reason

    # Denied: subcommands/chains trying to invoke forbidden binaries
    chained = "python -c 'import os' && git push"
    ok, reason = check_command_allowed(chained, backend_allowlist)
    assert ok is False
    assert "forbidden" in reason


def test_builtin_security_scanner_comprehensive(tmp_path: Path) -> None:
    scanner = BuiltinSecurityScanner()

    # 1. Secret detection
    leak_file = tmp_path / "credentials.py"
    leak_file.write_text("GITHUB_TOKEN = 'ghp_123456789012345678901234567890123456'\n", encoding="utf-8")
    assert len(scanner.scan_file_for_secrets(leak_file)) >= 1

    # 2. Dangerous AST detection
    eval_file = tmp_path / "unsafe_eval.py"
    eval_file.write_text("result = eval(user_input)\n", encoding="utf-8")
    findings = scanner.scan_file_for_dangerous_ast(eval_file)
    assert len(findings) == 1
    assert "eval()" in findings[0]

    exec_file = tmp_path / "unsafe_exec.py"
    exec_file.write_text("exec(dynamically_generated_code)\n", encoding="utf-8")
    findings = scanner.scan_file_for_dangerous_ast(exec_file)
    assert len(findings) == 1
    assert "exec()" in findings[0]

    import_file = tmp_path / "unsafe_import.py"
    import_file.write_text("mod = __import__('os')\n", encoding="utf-8")
    findings = scanner.scan_file_for_dangerous_ast(import_file)
    assert len(findings) == 1
    assert "__import__()" in findings[0]

    # Clean file produces 0 findings
    clean_file = tmp_path / "safe.py"
    clean_file.write_text("def safe_add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
    assert len(scanner.scan_file_for_dangerous_ast(clean_file)) == 0
    assert len(scanner.scan_file_for_secrets(clean_file)) == 0

    # 3. High-risk path changes
    flagged = scanner.check_high_risk_paths([
        "src/auth/token_service.py",
        "migrations/0001_initial.py",
        "payments/stripe_client.py",
        "requirements.txt",
        ".github/workflows/deploy.yml",
        "calc.py",
    ])
    assert "src/auth/token_service.py" in flagged
    assert "migrations/0001_initial.py" in flagged
    assert "payments/stripe_client.py" in flagged
    assert "requirements.txt" in flagged
    assert ".github/workflows/deploy.yml" in flagged
    assert "calc.py" not in flagged


def test_secret_redaction_comprehensive() -> None:
    raw_text = (
        "Output with api_key='sk-ant-api03-abcdef1234567890abcdef' "
        "and token: ghp_123456789012345678901234567890123456 "
        "and password = 'SuperSecretPassword123' "
        "and -----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----"
    )
    redacted = redact_secrets(raw_text)
    assert "sk-ant-api03" not in redacted
    assert "ghp_123456" not in redacted
    assert "SuperSecretPassword123" not in redacted
    assert "-----BEGIN RSA PRIVATE KEY-----" not in redacted
    assert "[REDACTED_SECRET]" in redacted


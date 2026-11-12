"""
devswarm.gates — Deterministic Quality Gates engine for pre-commit verification.

Stages:
  1. Compile / Syntax Check: bytecode compilation check (python -m py_compile {files}).
  2. Lint Check: static style check (ruff check {files}).
  3. Deterministic Test Suite: unit test execution (pytest -q).
  4. Security Scan:
     - Scan for secrets / tokens (regex matching).
     - AST analysis for dangerous builtins (eval, exec, __import__).
     - High-risk path detection (auth, payments, migrations, CI, dependencies).

If required gates pass:
  - If high-risk changes detected -> route to human approval (NEEDS_HUMAN).
  - Otherwise -> commit and merge.
If required gates fail:
  - Generate structured repair feedback for 1 automated agent retry turn.
  - If repair fails, abort task merge.
"""

from __future__ import annotations

import ast
import fnmatch
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

from devswarm.platform.events import SECRET_PATTERNS, redact_secrets, truncate_output
from devswarm.platform.sandbox import Sandbox, CommandResult

logger = logging.getLogger(__name__)

DEFAULT_GATES_CONFIG_PATH = Path("config/gates.yaml")
DEFAULT_POLICY_CONFIG_PATH = Path("config/policy.yaml")


@dataclass
class GateConfig:
    """Configuration for an individual deterministic quality gate."""
    name: str
    enabled: bool = True
    required: bool = True
    description: str = ""
    command: str = ""
    type: str = "command"
    timeout_s: float = 30.0


@dataclass
class GateResult:
    """Execution result for a single quality gate."""
    name: str
    stage: str
    passed: bool
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    error_message: str | None = None
    required: bool = True


@dataclass
class GateSuiteResult:
    """Aggregated outcome of running the deterministic quality gate suite."""
    passed: bool
    results: dict[str, GateResult] = field(default_factory=dict)
    duration_s: float = 0.0
    summary: str = ""
    feedback_prompt: str = ""
    has_high_risk_changes: bool = False
    high_risk_files: list[str] = field(default_factory=list)


# Alias for backward/forward compatibility
QualityGateResult = GateSuiteResult


class BuiltinSecurityScanner:
    """Static security analysis inspecting file diffs, secrets, and AST structures."""

    def __init__(self, high_risk_patterns: list[str] | None = None) -> None:
        self.high_risk_patterns = high_risk_patterns or [
            "*auth*", "*payment*", "*migration*", "*secret*", "*.github/workflows/*",
            "package-lock.json", "poetry.lock", "Pipfile.lock", "requirements.txt",
        ]

    def scan_file_for_secrets(self, path: Path) -> list[str]:
        """Scan file text for hardcoded API keys, private keys, or passwords."""
        if not path.is_file():
            return []
        findings: list[str] = []
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SECRET_PATTERNS:
                matches = pattern.findall(content)
                if matches:
                    findings.append(f"Hardcoded secret detected matching pattern: {pattern.pattern[:30]}...")
        except Exception as exc:
            findings.append(f"Error reading file for secret scan: {exc}")
        return findings

    def scan_file_for_dangerous_ast(self, path: Path) -> list[str]:
        """Parse Python AST to detect dangerous builtins like eval or exec."""
        if not path.is_file() or path.suffix != ".py":
            return []
        findings: list[str] = []
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ("eval", "exec"):
                            findings.append(
                                f"Disallowed function '{node.func.id}()' called at line {node.lineno}"
                            )
                        elif node.func.id == "__import__":
                            findings.append(
                                f"Dynamic import '__import__()' called at line {node.lineno}"
                            )
        except SyntaxError as exc:
            findings.append(f"Syntax error during AST security analysis: {exc}")
        except Exception as exc:
            findings.append(f"AST scan error: {exc}")
        return findings

    def check_high_risk_paths(self, modified_files: list[str]) -> list[str]:
        """Return list of modified files that match high-risk security patterns."""
        flagged: list[str] = []
        for file_str in modified_files:
            norm = file_str.strip("/\\")
            for pattern in self.high_risk_patterns:
                if fnmatch.fnmatch(norm, pattern) or fnmatch.fnmatch(Path(norm).name, pattern):
                    flagged.append(norm)
                    break
        return sorted(list(set(flagged)))


class GateRunner:
    """Manages the execution of the deterministic quality gate suite."""

    def __init__(
        self,
        sandbox: Sandbox,
        gates_config_path: str | Path = DEFAULT_GATES_CONFIG_PATH,
        policy_config_path: str | Path = DEFAULT_POLICY_CONFIG_PATH,
    ) -> None:
        self.sandbox = sandbox
        self.gates_config_path = Path(gates_config_path)
        self.policy_config_path = Path(policy_config_path)
        self.gates: dict[str, GateConfig] = {}
        self.high_risk_patterns: list[str] = []
        self._load_configs()
        self.security_scanner = BuiltinSecurityScanner(high_risk_patterns=self.high_risk_patterns)

    def _load_configs(self) -> None:
        """Load gates and security policy configurations."""
        if self.gates_config_path.exists():
            try:
                with open(self.gates_config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                gates_dict = data.get("gates", {})
                for name, g_data in gates_dict.items():
                    self.gates[name] = GateConfig(
                        name=name,
                        enabled=bool(g_data.get("enabled", True)),
                        required=bool(g_data.get("required", True)),
                        description=g_data.get("description", ""),
                        command=g_data.get("command", ""),
                        type=g_data.get("type", "command"),
                        timeout_s=float(g_data.get("timeout_s", 30.0)),
                    )
            except Exception as exc:
                logger.error("Failed to load gates config: %s", exc)

        # Fallback default gates if empty
        if not self.gates:
            self.gates = {
                "compile": GateConfig(
                    name="compile",
                    enabled=True,
                    required=True,
                    description="Compile syntax check",
                    command="python -m py_compile {files}",
                    timeout_s=15.0,
                ),
                "test": GateConfig(
                    name="test",
                    enabled=True,
                    required=True,
                    description="Pytest unit tests",
                    command="pytest -q",
                    timeout_s=30.0,
                ),
                "security": GateConfig(
                    name="security",
                    enabled=True,
                    required=True,
                    description="Static security scanner",
                    type="builtin_security_scan",
                    timeout_s=15.0,
                ),
            }

        if self.policy_config_path.exists():
            try:
                with open(self.policy_config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                self.high_risk_patterns = data.get("high_risk_paths", [])
            except Exception:
                pass

    async def run_suite(
        self,
        worktree_dir: Path | str,
        modified_files: list[str] | None = None,
        test_scope: list[str] | None = None,
    ) -> GateSuiteResult:
        """Execute all enabled gates in deterministic sequence."""
        start_time = time.monotonic()
        worktree_path = Path(worktree_dir).resolve()
        files = modified_files or []

        # Find python files among modified files
        py_files = [f for f in files if f.endswith(".py")]
        if not py_files:
            # Fallback: scan worktree for any python files
            py_files = [
                str(p.relative_to(worktree_path))
                for p in worktree_path.glob("**/*.py")
                if not any(part.startswith(".") for part in p.relative_to(worktree_path).parts)
            ]
        files_arg = " ".join(py_files) if py_files else "."

        suite_results: dict[str, GateResult] = {}
        all_passed = True
        failing_gates: list[str] = []

        # Check for high-risk path changes
        high_risk_flagged = self.security_scanner.check_high_risk_paths(files)
        has_high_risk = len(high_risk_flagged) > 0

        for gate_name, gate in self.gates.items():
            if not gate.enabled:
                continue

            gate_start = time.monotonic()

            if gate.type == "builtin_security_scan" or gate_name == "security":
                # Builtin Security Scan
                sec_errors: list[str] = []
                # Scan modified files
                target_paths = [worktree_path / f for f in files if (worktree_path / f).exists()]
                if not target_paths:
                    # If modified files not specified, scan python files in worktree root
                    target_paths = list(worktree_path.glob("*.py"))

                for p in target_paths:
                    sec_errors.extend(self.security_scanner.scan_file_for_secrets(p))
                    sec_errors.extend(self.security_scanner.scan_file_for_dangerous_ast(p))

                gate_duration = time.monotonic() - gate_start
                passed = len(sec_errors) == 0

                res = GateResult(
                    name=gate_name,
                    stage="security",
                    passed=passed,
                    exit_code=0 if passed else 1,
                    stdout="No security violations detected." if passed else "",
                    stderr="\n".join(sec_errors) if sec_errors else "",
                    duration_s=gate_duration,
                    error_message=None if passed else f"Security violations: {sec_errors[0]}",
                    required=gate.required,
                )
            else:
                # Command gate executed inside sandbox
                # If command requires {files} but no python files exist in worktree, pass automatically
                if "{files}" in gate.command and not py_files:
                    res = GateResult(
                        name=gate_name,
                        stage=gate_name,
                        passed=True,
                        exit_code=0,
                        stdout="No python files to verify for this gate.",
                        stderr="",
                        duration_s=0.0,
                        error_message=None,
                        required=gate.required,
                    )
                    suite_results[gate_name] = res
                    continue

                cmd = gate.command.replace("{files}", files_arg)
                if gate_name == "test" and test_scope:
                    # Target specific test files/directories relevant to current task
                    cmd = f"{gate.command.replace('{files}', '').strip()} {' '.join(test_scope)}".strip()
                res_cmd = self.sandbox.run_command(
                    command=cmd,
                    worktree_dir=worktree_path,
                    timeout=gate.timeout_s,
                )

                gate_duration = time.monotonic() - gate_start
                passed = res_cmd.exit_code == 0

                # Special case for test gate (pytest): exit code 5 means no tests were collected.
                # If no test files exist in the worktree yet (e.g. backend building initial files), it's not a failure.
                if gate_name == "test" and res_cmd.exit_code == 5:
                    test_files = list(worktree_path.glob("**/test_*.py")) + list(worktree_path.glob("**/*_test.py"))
                    if not test_files:
                        passed = True

                error_msg = None if passed else f"Command '{cmd}' failed with exit code {res_cmd.exit_code}"

                res = GateResult(
                    name=gate_name,
                    stage=gate_name,
                    passed=passed,
                    exit_code=res_cmd.exit_code,
                    stdout=redact_secrets(res_cmd.stdout),
                    stderr=redact_secrets(res_cmd.stderr),
                    duration_s=gate_duration,
                    error_message=error_msg,
                    required=gate.required,
                )

            suite_results[gate_name] = res

            if gate.required and not res.passed:
                all_passed = False
                failing_gates.append(gate_name)

        total_duration = time.monotonic() - start_time

        # Formulate feedback prompt for agent repair
        feedback_prompt = ""
        if not all_passed:
            feedback_parts = [
                "❌ DETERMINISTIC QUALITY GATES FAILED",
                "Your changes must pass all required quality gates before merging. Please repair the following issues:\n",
            ]
            for fg in failing_gates:
                r = suite_results[fg]
                feedback_parts.append(f"--- [Gate: {fg.upper()}] ---")
                if r.error_message:
                    feedback_parts.append(f"Error: {r.error_message}")
                if r.stderr.strip():
                    feedback_parts.append(f"STDERR:\n{r.stderr.strip()}")
                if r.stdout.strip():
                    feedback_parts.append(f"STDOUT:\n{r.stdout.strip()}")
                feedback_parts.append("")

            feedback_parts.append("Inspect the files, fix the syntax/test/security errors, and call finish_task when resolved.")
            feedback_prompt = "\n".join(feedback_parts)

        summary_status = "PASSED" if all_passed else f"FAILED ({', '.join(failing_gates)})"
        summary = f"Quality Gates {summary_status} in {total_duration:.2f}s"

        return GateSuiteResult(
            passed=all_passed,
            results=suite_results,
            duration_s=total_duration,
            summary=summary,
            feedback_prompt=feedback_prompt,
            has_high_risk_changes=has_high_risk,
            high_risk_files=high_risk_flagged,
        )


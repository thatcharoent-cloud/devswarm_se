"""
devswarm.self_healing — Deterministic failure diagnostics, attempt tracking, and cloud escalation.

Features:
  1. extract_failure_diagnostics: Pure Python parser that extracts failing test names,
     assertions, line numbers, and truncated stack traces from gate outputs.
  2. TaskAttemptManager: Append-only attempt history per task (never overwritten).
  3. CloudEscalationHandler: Optional escalation (--escalate-to-cloud) to a cloud debugger
     after MAX_FIX_ATTEMPTS local failures, subject to OutboundFilter and Cloud USD budget.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devswarm.gates import QualityGateResult
from devswarm.platform.budget import CloudBudgetTracker
from devswarm.platform.outbound_filter import OutboundFilter
from devswarm.providers.base import BaseProvider, ChatMessage

logger = logging.getLogger(__name__)


@dataclass
class TaskAttempt:
    """Immutable record of an individual task implementation attempt."""
    attempt_number: int
    timestamp: str
    files_modified: list[str]
    gate_passed: bool
    gate_summary: str
    diagnostics: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "timestamp": self.timestamp,
            "files_modified": self.files_modified,
            "gate_passed": self.gate_passed,
            "gate_summary": self.gate_summary,
            "diagnostics": self.diagnostics,
        }


def extract_failure_diagnostics(gate_result: QualityGateResult, max_chars: int = 2500) -> str:
    """
    Deterministically extracts failing test names, line numbers, and truncated
    stack traces from QualityGateResult without any LLM calls.
    """
    lines: list[str] = []

    # 1. Check compile gate
    compile_gate = gate_result.results.get("compile")
    if compile_gate and not compile_gate.passed:
        lines.append("### Compilation / Syntax Failure:")
        msg = (compile_gate.stderr or compile_gate.stdout or compile_gate.error_message or "").strip()
        lines.append(msg[:1000])

    # 2. Check test gate (pytest output)
    test_gate = gate_result.results.get("test")
    if test_gate and not test_gate.passed:
        combined = (test_gate.stdout + "\n" + test_gate.stderr).strip()

        lines.append("### Automated Test Failures:")

        # Extract failed test identifiers: e.g. "FAILED tests/test_calc.py::test_add - AssertionError..."
        failed_patterns = re.findall(r"(FAILED\s+[^\n]+)", combined)
        if failed_patterns:
            lines.append("Failing tests:")
            for p in failed_patterns[:8]:
                lines.append(f"- {p.strip()}")

        # Extract assertion and error blocks (E   AssertionError: ...)
        error_lines: list[str] = []
        for line in combined.splitlines():
            if line.startswith("E   ") or "AssertionError" in line or "Error:" in line:
                error_lines.append(line.strip())

        if error_lines:
            lines.append("\nFailure details:")
            for err in error_lines[:10]:
                lines.append(f"  {err}")

        # Extract stack frames showing file and line
        frames = re.findall(r"((?:tests/|[a-zA-Z0-9_\.]+\.py:)[^:\n]+:\d+:[^\n]*)", combined)
        if frames:
            lines.append("\nFailure locations:")
            for frame in frames[:6]:
                lines.append(f"  at {frame.strip()}")

        if not failed_patterns and not error_lines:
            # Fallback to truncated raw output
            lines.append(combined[-1500:])

    # 3. Check lint gate
    lint_gate = gate_result.results.get("lint")
    if lint_gate and not lint_gate.passed and not lines:
        lines.append("### Linting Issues:")
        lint_out = (lint_gate.stdout or lint_gate.stderr).strip()
        lines.append(lint_out[:1000])

    diag_text = "\n".join(lines).strip()
    if len(diag_text) > max_chars:
        diag_text = diag_text[:max_chars] + "\n... [truncated]"

    return (
        f"<untrusted_gate_feedback>\n"
        f"The previous implementation failed deterministic quality gates.\n\n"
        f"{diag_text}\n\n"
        f"Fix the code to resolve these specific errors and make all tests pass.\n"
        f"</untrusted_gate_feedback>"
    )


class TaskAttemptManager:
    """Tracks full attempt history per task across fix loops."""

    def __init__(self) -> None:
        self.attempts_by_task: dict[str, list[TaskAttempt]] = {}

    def record_attempt(
        self,
        task_id: str,
        files_modified: list[str],
        gate_passed: bool,
        gate_summary: str,
        diagnostics: str = "",
    ) -> TaskAttempt:
        """Append a new attempt record without overwriting previous history."""
        if task_id not in self.attempts_by_task:
            self.attempts_by_task[task_id] = []

        history = self.attempts_by_task[task_id]
        attempt_num = len(history) + 1

        attempt = TaskAttempt(
            attempt_number=attempt_num,
            timestamp=datetime.now(timezone.utc).isoformat(),
            files_modified=list(files_modified),
            gate_passed=gate_passed,
            gate_summary=gate_summary,
            diagnostics=diagnostics,
        )
        history.append(attempt)
        return attempt

    def get_attempts(self, task_id: str) -> list[TaskAttempt]:
        return list(self.attempts_by_task.get(task_id, []))

    def attempt_count(self, task_id: str) -> int:
        return len(self.attempts_by_task.get(task_id, []))


class CloudEscalationHandler:
    """Handles optional escalation to a cloud debugger role."""

    def __init__(
        self,
        enabled: bool = False,
        provider: BaseProvider | None = None,
        model: str = "claude-3-5-haiku",
        outbound_filter: OutboundFilter | None = None,
        budget_tracker: CloudBudgetTracker | None = None,
    ) -> None:
        self.enabled = enabled
        self.provider = provider
        self.model = model
        self.outbound_filter = outbound_filter
        self.budget_tracker = budget_tracker

    async def request_debug_advice(
        self,
        task_id: str,
        task_title: str,
        write_scope: list[str],
        diagnostics: str,
        recent_code: str = "",
        run_id: str = "",
    ) -> str | None:
        """
        Consult cloud debugger if enabled.
        Returns suggested patch guidance for the local coder.
        """
        if not self.enabled or not self.provider:
            return None

        # 1. Estimate and reserve cost
        est_cost = 0.05
        res_id = ""
        if self.budget_tracker:
            est_cost = self.budget_tracker.estimate_cost(self.model, input_tokens=3000, max_tokens=1500)
            res_id = self.budget_tracker.reserve(
                role="debugger",
                provider=self.provider.name,
                model=self.model,
                estimated_cost=est_cost,
            )

        system_prompt = (
            "You are a Senior Debugging Specialist. A local coding agent has failed multiple fix attempts. "
            "Your job is to diagnose the root cause and provide concise, targeted patch suggestions strictly "
            f"within the task write scope: {', '.join(write_scope)}.\n"
            "RULES:\n"
            "1. Be extremely concise.\n"
            "2. Explain the root cause of the test failure.\n"
            "3. Provide the corrected Python snippet for the local coder to apply.\n"
        )

        user_content = (
            f"Task: [{task_id}] {task_title}\n"
            f"Write Scope: {', '.join(write_scope)}\n\n"
            f"Deterministic Failure Diagnostics:\n{diagnostics}\n\n"
        )
        if recent_code:
            user_content += f"Current Code:\n```python\n{recent_code}\n```\n"

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_content),
        ]

        # 2. Pass through OutboundFilter
        if self.outbound_filter:
            messages = self.outbound_filter.filter_and_validate(
                role="reviewer",  # Reviewer has diffs/logs permission
                provider=self.provider.name,
                model=self.model,
                messages=messages,
                content_classes=["gate_logs", "test_files"],
                run_id=run_id,
            )

        # 3. Call cloud model
        resp = await self.provider.chat(
            model=self.model,
            messages=messages,
            max_tokens=1500,
            temperature=0.1,
        )

        # 4. Commit actual cost
        if self.budget_tracker and res_id:
            self.budget_tracker.commit(res_id, resp.cost_usd)

        advice_block = (
            f"<cloud_debugger_guidance>\n"
            f"Senior Debugger Diagnosis ({self.provider.name}/{self.model}):\n"
            f"{resp.content}\n"
            f"</cloud_debugger_guidance>"
        )
        return advice_block


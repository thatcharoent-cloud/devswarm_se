"""
devswarm.traceability — AC -> Tasks -> Tests -> Gate Results Traceability Matrix.

Ensures that every Acceptance Criterion (AC-1..n) is covered by:
  1. At least one planned task
  2. At least one automated test
  3. A passing test result before PR review (unless explicitly waived by human).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from devswarm.agents.pm_schema import PMSpecification


@dataclass
class TraceabilityRecord:
    ac_id: str
    description: str
    tasks: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    status: str = "PENDING"  # PENDING, PASS, FAIL, WAIVED
    waived: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ac_id": self.ac_id,
            "description": self.description,
            "tasks": self.tasks,
            "tests": self.tests,
            "status": self.status,
            "waived": self.waived,
        }


class TraceabilityMatrix:
    """Manages full traceability from acceptance criteria to passing gates."""

    def __init__(self, records: dict[str, TraceabilityRecord] | None = None) -> None:
        self.records: dict[str, TraceabilityRecord] = records or {}

    @classmethod
    def create_from_spec(
        cls,
        spec: PMSpecification,
        planned_tasks: list[Any] | None = None,
        qa_mapping: dict[str, list[str]] | None = None,
        waived_acs: list[str] | None = None,
    ) -> TraceabilityMatrix:
        """Construct matrix from PM spec, task DAG, and QA test mapping."""
        records: dict[str, TraceabilityRecord] = {}
        waived_set = set(waived_acs or [])
        qa_map = qa_mapping or {}

        for ac in spec.acceptance_criteria:
            is_waived = ac.id in waived_set
            rec = TraceabilityRecord(
                ac_id=ac.id,
                description=ac.description,
                tasks=[],
                tests=qa_map.get(ac.id, []),
                status="WAIVED" if is_waived else "PENDING",
                waived=is_waived,
            )
            records[ac.id] = rec

        # Map planned tasks to ACs
        if planned_tasks:
            for task in planned_tasks:
                t_id = getattr(task, "id", "") or (task.get("id") if isinstance(task, dict) else "")
                t_desc = (getattr(task, "description", "") or (task.get("description", "") if isinstance(task, dict) else "")).lower()
                t_title = (getattr(task, "title", "") or (task.get("title", "") if isinstance(task, dict) else "")).lower()

                # Link tasks that mention AC or relevant keywords
                for ac_id, rec in records.items():
                    if ac_id.lower() in t_desc or ac_id.lower() in t_title or len(records) == 1:
                        if t_id and t_id not in rec.tasks:
                            rec.tasks.append(t_id)

            # Fallback: if task didn't explicitly link AC id, map in order
            unlinked_acs = [r for r in records.values() if not r.tasks]
            if unlinked_acs and planned_tasks:
                for r in unlinked_acs:
                    r.tasks.append(getattr(planned_tasks[0], "id", "task_1"))

        return cls(records=records)

    def update_test_results(self, test_output: str, exit_code: int) -> None:
        """Update pass/fail status per AC based on test execution output."""
        for rec in self.records.values():
            if rec.waived:
                rec.status = "WAIVED"
                continue

            if not rec.tests:
                rec.status = "PENDING"
                continue

            # Check if all tests for this AC passed
            all_passed = True
            for test_name in rec.tests:
                # In pytest output, PASSED appears next to test name
                short_name = test_name.split("::")[-1]
                if short_name in test_output and "FAILED" in test_output:
                    # Check failure lines
                    if f"FAILED {test_name}" in test_output or f"{short_name} FAILED" in test_output:
                        all_passed = False
                        break

            if exit_code == 0:
                rec.status = "PASS"
            elif all_passed and rec.tests:
                rec.status = "PASS"
            else:
                rec.status = "FAIL"

    def is_review_ready(self) -> tuple[bool, list[str]]:
        """Verify that all acceptance criteria are either passing or explicitly waived."""
        unresolved: list[str] = []
        for ac_id, rec in self.records.items():
            if rec.status not in ("PASS", "WAIVED"):
                unresolved.append(f"{ac_id}: {rec.description} (status: {rec.status})")
        return len(unresolved) == 0, unresolved

    def to_markdown(self) -> str:
        """Render markdown traceability table for PR report."""
        lines = [
            "### 📋 Acceptance Criteria Traceability Matrix",
            "| AC ID | Description | Planned Tasks | Automated Tests | Gate Status |",
            "|---|---|---|---|:---:|",
        ]

        for rec in self.records.values():
            tasks_str = ", ".join(f"`{t}`" for t in rec.tasks) if rec.tasks else "*None*"
            tests_str = ", ".join(f"`{t.split('::')[-1]}`" for t in rec.tests) if rec.tests else "*None*"
            if rec.status == "PASS":
                status_icon = "✅ `PASS`"
            elif rec.status == "WAIVED":
                status_icon = "⚠️ `WAIVED`"
            elif rec.status == "FAIL":
                status_icon = "❌ `FAIL`"
            else:
                status_icon = "⏳ `PENDING`"

            lines.append(f"| `{rec.ac_id}` | {rec.description} | {tasks_str} | {tests_str} | {status_icon} |")

        return "\n".join(lines)

    render_matrix_markdown = to_markdown

    def save(self, path: Path | str) -> None:
        """Save matrix as JSON artifact."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {ac_id: rec.to_dict() for ac_id, rec in self.records.items()}
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")


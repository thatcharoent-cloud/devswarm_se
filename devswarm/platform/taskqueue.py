"""
devswarm.platform.taskqueue — SQLite-backed DAG task queue for devswarm.

Features:
  - States: pending, ready, running, gates, review, done, failed, blocked, needs_human.
  - DAG dependency resolution: automatically triggers ready tasks upon dependency completion.
  - Failure propagation: blocks dependent tasks if an upstream dependency fails.
  - Crash recovery: resets abandoned 'running' or 'gates' tasks back to 'ready' with retry counting.
  - Concurrency management: yields next ready tasks up to concurrency limits.
  - Thread-safe & process-persisted in SQLite (.swarm/runs/<run_id>/tasks.db).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    PENDING = "pending"          # Waiting on upstream task dependencies
    READY = "ready"              # Dependencies satisfied, ready for worker
    RUNNING = "running"          # Currently executing specialist tool loop
    GATES = "gates"              # Running deterministic quality gates
    REVIEW = "review"            # Gates passed, waiting for reviewer
    DONE = "done"                # Fully completed and committed/merged
    FAILED = "failed"            # Failed execution or gates exhausted
    BLOCKED = "blocked"          # Upstream dependency failed
    NEEDS_HUMAN = "needs_human"  # Merge conflict or security policy escalation


@dataclass
class SwarmTask:
    """Represents a discrete engineering task in the swarm."""
    id: str
    title: str
    role: str
    description: str
    acceptance_criteria: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    write_scope: list[str] = field(default_factory=list)
    read_scope: list[str] = field(default_factory=list)
    risk_level: str = "low"     # "low", "medium", "high"
    minutes_budget: float = 10.0
    state: TaskState = TaskState.PENDING
    retry_count: int = 0
    max_retries: int = 2
    error: str = ""
    result_summary: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "role": self.role,
            "description": self.description,
            "acceptance_criteria": self.acceptance_criteria,
            "depends_on": self.depends_on,
            "write_scope": self.write_scope,
            "read_scope": self.read_scope,
            "risk_level": self.risk_level,
            "minutes_budget": self.minutes_budget,
            "state": self.state.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "error": self.error,
            "result_summary": self.result_summary,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class TaskQueue:
    """SQLite-backed queue managing task DAG state transitions."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create task queue schema if it does not exist."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    role TEXT NOT NULL,
                    description TEXT NOT NULL,
                    acceptance_criteria TEXT NOT NULL,
                    depends_on TEXT NOT NULL,
                    write_scope TEXT NOT NULL,
                    read_scope TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    minutes_budget REAL NOT NULL,
                    state TEXT NOT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 2,
                    error TEXT NOT NULL DEFAULT '',
                    result_summary TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_state ON tasks(state)")

    def add_tasks(self, tasks: list[SwarmTask]) -> None:
        """Add tasks to queue. If a task has no dependencies, it begins as READY."""
        with self._get_conn() as conn:
            for t in tasks:
                row = conn.execute("SELECT state FROM tasks WHERE id = ?", (t.id,)).fetchone()
                if row and row["state"] == TaskState.DONE.value:
                    continue
                init_state = TaskState.READY if not t.depends_on else TaskState.PENDING
                t.state = init_state
                conn.execute("""
                    INSERT OR REPLACE INTO tasks (
                        id, title, role, description, acceptance_criteria,
                        depends_on, write_scope, read_scope, risk_level,
                        minutes_budget, state, retry_count, max_retries,
                        error, result_summary, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    t.id, t.title, t.role, t.description,
                    json.dumps(t.acceptance_criteria),
                    json.dumps(t.depends_on),
                    json.dumps(t.write_scope),
                    json.dumps(t.read_scope),
                    t.risk_level, t.minutes_budget,
                    t.state.value, t.retry_count, t.max_retries,
                    t.error, t.result_summary, t.created_at, t.updated_at,
                ))
        self.resolve_dependencies()

    def get_task(self, task_id: str) -> SwarmTask | None:
        """Fetch a task by id."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if not row:
                return None
            return self._row_to_task(row)

    def get_all_tasks(self) -> list[SwarmTask]:
        """Fetch all tasks in queue."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM tasks ORDER BY created_at ASC").fetchall()
            return [self._row_to_task(r) for r in rows]

    def set_state(
        self,
        task_id: str,
        state: TaskState,
        error: str = "",
        result_summary: str = "",
    ) -> None:
        """Update task state and resolve dependent DAG transitions."""
        now = time.time()
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE tasks
                SET state = ?, error = ?, result_summary = ?, updated_at = ?
                WHERE id = ?
            """, (state.value, error, result_summary, now, task_id))

        # Re-resolve dependencies when a task reaches a terminal state
        if state in (TaskState.DONE, TaskState.FAILED, TaskState.BLOCKED, TaskState.NEEDS_HUMAN):
            self.resolve_dependencies()

    def resolve_dependencies(self) -> None:
        """Evaluate all PENDING tasks and transition them to READY or BLOCKED based on dependencies."""
        tasks = self.get_all_tasks()
        task_map = {t.id: t for t in tasks}

        for t in tasks:
            if t.state != TaskState.PENDING:
                continue

            deps = [task_map.get(dep_id) for dep_id in t.depends_on]
            # If any dependency is failed/blocked/needs_human -> this task becomes BLOCKED
            if any(d and d.state in (TaskState.FAILED, TaskState.BLOCKED, TaskState.NEEDS_HUMAN) for d in deps):
                self.set_state(
                    t.id,
                    TaskState.BLOCKED,
                    error=f"Blocked by upstream task failure in {[d.id for d in deps if d and d.state in (TaskState.FAILED, TaskState.BLOCKED, TaskState.NEEDS_HUMAN)]}",
                )
                continue

            # If all dependencies are DONE -> this task becomes READY
            if all(d and d.state == TaskState.DONE for d in deps):
                self.set_state(t.id, TaskState.READY)

    def get_next_ready(self, concurrency_limit: int = 1) -> list[SwarmTask]:
        """Return ready tasks respecting current running/active task concurrency."""
        with self._get_conn() as conn:
            active_count = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE state IN (?, ?)",
                (TaskState.RUNNING.value, TaskState.GATES.value)
            ).fetchone()[0]

            available_slots = max(0, concurrency_limit - active_count)
            if available_slots <= 0:
                return []

            rows = conn.execute(
                "SELECT * FROM tasks WHERE state = ? ORDER BY created_at ASC LIMIT ?",
                (TaskState.READY.value, available_slots)
            ).fetchall()

            return [self._row_to_task(r) for r in rows]

    def recover_after_crash(self) -> int:
        """Reset abandoned tasks in RUNNING or GATES state back to READY (or FAILED if retries exhausted)."""
        tasks = self.get_all_tasks()
        recovered = 0
        now = time.time()

        for t in tasks:
            if t.state in (TaskState.RUNNING, TaskState.GATES):
                recovered += 1
                new_retry = t.retry_count + 1
                if new_retry <= t.max_retries:
                    new_state = TaskState.READY
                    err_msg = f"Crash recovery: resumed from {t.state.value} (attempt {new_retry}/{t.max_retries})"
                else:
                    new_state = TaskState.FAILED
                    err_msg = f"Crash recovery: exhausted retries ({new_retry}/{t.max_retries})"

                with self._get_conn() as conn:
                    conn.execute("""
                        UPDATE tasks
                        SET state = ?, retry_count = ?, error = ?, updated_at = ?
                        WHERE id = ?
                    """, (new_state.value, new_retry, err_msg, now, t.id))

        if recovered > 0:
            logger.info("Recovered %d in-flight tasks after process restart", recovered)
            self.resolve_dependencies()

        return recovered

    def is_finished(self) -> bool:
        """Return True when no tasks are PENDING, READY, RUNNING, or in GATES."""
        with self._get_conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE state IN (?, ?, ?, ?, ?)",
                (
                    TaskState.PENDING.value,
                    TaskState.READY.value,
                    TaskState.RUNNING.value,
                    TaskState.GATES.value,
                    TaskState.REVIEW.value,
                )
            ).fetchone()[0]
            return count == 0

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> SwarmTask:
        return SwarmTask(
            id=row["id"],
            title=row["title"],
            role=row["role"],
            description=row["description"],
            acceptance_criteria=json.loads(row["acceptance_criteria"]),
            depends_on=json.loads(row["depends_on"]),
            write_scope=json.loads(row["write_scope"]),
            read_scope=json.loads(row["read_scope"]),
            risk_level=row["risk_level"],
            minutes_budget=row["minutes_budget"],
            state=TaskState(row["state"]),
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
            error=row["error"],
            result_summary=row["result_summary"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


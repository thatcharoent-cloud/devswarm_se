"""
devswarm.agents.plan_schema — Pydantic plan models and deterministic DAG validation.

Validation rules:
  1. Every task's role must exist and be enabled.
  2. The task DAG must be strictly acyclic (no circular dependencies).
  3. At most MAX_FILES_PER_TASK patterns per task's write_scope.
  4. Parallel-capable tasks must NOT have overlapping write scopes.
  5. Total tasks must not exceed MAX_SUBTASKS.
  6. Total estimated minutes must fit within MAX_RUN_MINUTES.
"""

from __future__ import annotations

import fnmatch
from typing import Any
from pydantic import BaseModel, Field


class TaskPlanItem(BaseModel):
    """Specification for a single engineering subtask produced by the orchestrator."""
    id: str = Field(description="Unique task identifier, e.g. task-1, task-2")
    title: str = Field(description="Concise action-oriented summary of the task")
    role: str = Field(description="Specialist role: backend, tester, architect, frontend, security, devops")
    description: str = Field(description="Detailed instructions and implementation requirements")
    acceptance_criteria: list[str] = Field(description="Measurable conditions for task verification")
    depends_on: list[str] = Field(default_factory=list, description="IDs of tasks that must be completed first")
    write_scope: list[str] = Field(default_factory=list, description="File globs this task is permitted to modify")
    read_scope: list[str] = Field(default_factory=list, description="File globs this task may consult")
    risk_level: str = Field(default="low", description="Risk assessment: low, medium, or high")
    minutes_budget: float = Field(default=5.0, description="Estimated minutes allocated for this task")


class EngineeringPlan(BaseModel):
    """Structured engineering plan decomposing a user goal into an executable task DAG."""
    goal: str = Field(description="The primary user goal or feature specification")
    assumptions: list[str] = Field(default_factory=list, description="Technical assumptions made during planning")
    tasks: list[TaskPlanItem] = Field(description="Ordered list of specialist tasks forming a valid DAG")


class PlanValidationError(ValueError):
    """Raised when an engineering plan violates deterministic code constraints."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def check_acyclic_dag(tasks: list[TaskPlanItem]) -> tuple[bool, str]:
    """Verify that task dependencies form a strictly acyclic directed graph (DAG)."""
    task_ids = {t.id for t in tasks}
    adj: dict[str, list[str]] = {t.id: [] for t in tasks}

    for t in tasks:
        for dep in t.depends_on:
            if dep not in task_ids:
                return False, f"Task '{t.id}' references nonexistent dependency '{dep}'"
            adj[dep].append(t.id)

    # Cycle detection via DFS
    visited: dict[str, int] = {t_id: 0 for t_id in task_ids}  # 0: unvisited, 1: visiting, 2: visited

    def has_cycle(node: str) -> bool:
        visited[node] = 1
        for neighbor in adj[node]:
            if visited[neighbor] == 1:
                return True
            if visited[neighbor] == 0:
                if has_cycle(neighbor):
                    return True
        visited[node] = 2
        return False

    for t_id in task_ids:
        if visited[t_id] == 0:
            if has_cycle(t_id):
                return False, f"Cycle detected in task dependencies involving '{t_id}'"

    return True, ""


def find_ancestors(
    task_id: str,
    task_map: dict[str, TaskPlanItem],
    memo: dict[str, set[str]],
    visited: set[str] | None = None,
) -> set[str]:
    """Find all transitive upstream dependencies of task_id."""
    if task_id in memo:
        return memo[task_id]
    if visited is None:
        visited = set()
    if task_id in visited:
        return set()
    visited.add(task_id)

    ancestors: set[str] = set()
    task = task_map.get(task_id)
    if task:
        for dep in task.depends_on:
            ancestors.add(dep)
            ancestors.update(find_ancestors(dep, task_map, memo, visited.copy()))
    memo[task_id] = ancestors
    return ancestors


def normalize_plan_dependencies(tasks: list[TaskPlanItem]) -> None:
    """Tolerantly sequence tasks with overlapping write scopes if model omitted depends_on."""
    task_map = {t.id: t for t in tasks}
    memo: dict[str, set[str]] = {}

    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            t1 = tasks[i]
            t2 = tasks[j]
            ancestors_t1 = find_ancestors(t1.id, task_map, memo)
            ancestors_t2 = find_ancestors(t2.id, task_map, memo)
            if t2.id in ancestors_t1 or t1.id in ancestors_t2:
                continue

            overlaps = False
            for pat1 in t1.write_scope:
                for pat2 in t2.write_scope:
                    if pat1 == pat2 or fnmatch.fnmatch(pat1, pat2) or fnmatch.fnmatch(pat2, pat1):
                        overlaps = True
                        break
                if overlaps:
                    break

            # Check read-write dependencies (t2 reads what t1 creates)
            if not overlaps:
                for w in t1.write_scope:
                    for r in t2.read_scope:
                        if w == r or fnmatch.fnmatch(w, r) or fnmatch.fnmatch(r, w):
                            overlaps = True
                            break
                    if overlaps:
                        break

            # If t2 is a tester task with no dependencies and t1 is backend, link them
            if not overlaps and t1.role == "backend" and t2.role == "tester" and not t2.depends_on:
                overlaps = True

            if overlaps:
                # Later task in plan list depends on earlier task
                if t1.id not in t2.depends_on:
                    t2.depends_on.append(t1.id)
                    memo.clear()


def check_parallel_scope_overlap(tasks: list[TaskPlanItem]) -> list[str]:
    """Ensure parallel-capable tasks (no dependency relationship) do not overlap in write scope."""
    errors: list[str] = []
    task_map = {t.id: t for t in tasks}
    memo: dict[str, set[str]] = {}

    # Build ancestor sets
    ancestor_map = {t.id: find_ancestors(t.id, task_map, memo) for t in tasks}

    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            t1 = tasks[i]
            t2 = tasks[j]

            # Check if t1 and t2 have a dependency relationship
            if t2.id in ancestor_map[t1.id] or t1.id in ancestor_map[t2.id]:
                # Sequentially ordered by dependency -> write scopes may touch
                continue

            # Parallel-capable: check if write scopes overlap
            for pat1 in t1.write_scope:
                for pat2 in t2.write_scope:
                    # Direct match or glob overlap
                    if pat1 == pat2 or fnmatch.fnmatch(pat1, pat2) or fnmatch.fnmatch(pat2, pat1):
                        errors.append(
                            f"Parallel tasks '{t1.id}' and '{t2.id}' have overlapping write scope: '{pat1}' / '{pat2}'. "
                            f"Either add a dependency between them or isolate their write scopes."
                        )

    return errors


def validate_engineering_plan(
    plan: EngineeringPlan,
    enabled_roles: set[str] | list[str],
    max_files_per_task: int = 2,
    max_subtasks: int = 10,
    max_run_minutes: float = 45.0,
) -> None:
    """Perform deterministic validation on the generated plan."""
    errors: list[str] = []
    enabled = set(enabled_roles)

    if not plan.tasks:
        errors.append("Plan contains no tasks")
        raise PlanValidationError(errors)

    if len(plan.tasks) > max_subtasks:
        errors.append(f"Plan task count ({len(plan.tasks)}) exceeds MAX_SUBTASKS ({max_subtasks})")

    # 1. Role validation & write-scope item count
    total_minutes = 0.0
    for t in plan.tasks:
        if t.role not in enabled:
            errors.append(f"Task '{t.id}' assigns role '{t.role}' which is disabled or nonexistent (enabled: {sorted(enabled)})")

        if len(t.write_scope) > max_files_per_task:
            errors.append(
                f"Task '{t.id}' write_scope has {len(t.write_scope)} items, exceeding MAX_FILES_PER_TASK ({max_files_per_task})"
            )

        if len(t.acceptance_criteria) > 5:
            errors.append(
                f"Task '{t.id}' acceptance_criteria has {len(t.acceptance_criteria)} lines, exceeding limit of 5 lines"
            )

        total_minutes += t.minutes_budget

    if total_minutes > max_run_minutes:
        errors.append(
            f"Plan total minutes budget ({total_minutes:.1f}m) exceeds MAX_RUN_MINUTES ({max_run_minutes:.1f}m)"
        )

    # 2. DAG Acyclicity
    is_dag, dag_err = check_acyclic_dag(plan.tasks)
    if not is_dag:
        errors.append(dag_err)

    # 3. Parallel scope overlap (only check if DAG is valid)
    if is_dag:
        overlap_errs = check_parallel_scope_overlap(plan.tasks)
        errors.extend(overlap_errs)

    if errors:
        raise PlanValidationError(errors)


def build_plan_repair_prompt(original_plan_json: str, errors: list[str]) -> str:
    """Format validation errors into an actionable repair prompt for the orchestrator."""
    err_list = "\n".join(f"- {e}" for e in errors)
    return (
        "Your previously generated engineering plan failed deterministic code validation:\n"
        f"{err_list}\n\n"
        "Please fix the plan and return a valid JSON object matching the schema. "
        "Remember:\n"
        "1. All tasks must use enabled roles.\n"
        "2. The task graph must be strictly acyclic.\n"
        "3. Each task must have at most 2 write_scope entries.\n"
        "4. Each task must have at most 5 lines of acceptance criteria.\n"
        "5. Parallel-capable tasks must not have overlapping write scopes.\n"
        f"Original attempted plan:\n{original_plan_json}\n"
    )

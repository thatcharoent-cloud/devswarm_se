"""
tests/devswarm/test_plan_schema.py — Unit tests for plan Pydantic schema and deterministic DAG validation.
"""

import pytest
from devswarm.agents.plan_schema import (
    EngineeringPlan,
    TaskPlanItem,
    PlanValidationError,
    validate_engineering_plan,
    build_plan_repair_prompt,
)


def test_valid_plan_passes():
    plan = EngineeringPlan(
        goal="Add safe divide function",
        assumptions=["Use standard library math"],
        tasks=[
            TaskPlanItem(
                id="task-1",
                title="Implement safe divide",
                role="backend",
                description="Add divide check",
                acceptance_criteria=["Raises ValueError on 0", "Returns float"],
                write_scope=["src/calc.py"],
                minutes_budget=5.0,
            ),
            TaskPlanItem(
                id="task-2",
                title="Add unit tests",
                role="tester",
                description="Write pytest tests",
                acceptance_criteria=["Assert ValueError", "Assert normal divide"],
                depends_on=["task-1"],
                write_scope=["tests/test_calc.py"],
                minutes_budget=4.0,
            ),
        ],
    )

    validate_engineering_plan(
        plan,
        enabled_roles={"backend", "tester"},
        max_files_per_task=2,
        max_run_minutes=45.0,
    )


def test_cycle_in_dag_rejected():
    plan = EngineeringPlan(
        goal="Cyclic goal",
        tasks=[
            TaskPlanItem(
                id="task-1",
                title="Task 1",
                role="backend",
                description="desc 1",
                acceptance_criteria=["crit 1"],
                depends_on=["task-2"],  # Depends on task-2
                write_scope=["src/a.py"],
            ),
            TaskPlanItem(
                id="task-2",
                title="Task 2",
                role="backend",
                description="desc 2",
                acceptance_criteria=["crit 2"],
                depends_on=["task-1"],  # Depends on task-1 -> Cycle!
                write_scope=["src/b.py"],
            ),
        ],
    )

    with pytest.raises(PlanValidationError) as exc_info:
        validate_engineering_plan(plan, enabled_roles={"backend"})

    assert "Cycle detected" in str(exc_info.value)


def test_disabled_role_rejected():
    plan = EngineeringPlan(
        goal="Frontend goal",
        tasks=[
            TaskPlanItem(
                id="task-1",
                title="Build React button",
                role="frontend",  # Frontend disabled by default
                description="desc",
                acceptance_criteria=["crit"],
                write_scope=["src/button.tsx"],
            )
        ],
    )

    with pytest.raises(PlanValidationError) as exc_info:
        validate_engineering_plan(plan, enabled_roles={"backend", "tester"})

    assert "disabled or nonexistent" in str(exc_info.value)


def test_write_scope_item_cap_rejected():
    plan = EngineeringPlan(
        goal="Too many files",
        tasks=[
            TaskPlanItem(
                id="task-1",
                title="Modify 3 files",
                role="backend",
                description="desc",
                acceptance_criteria=["crit"],
                write_scope=["src/a.py", "src/b.py", "src/c.py"],  # 3 items > max 2!
            )
        ],
    )

    with pytest.raises(PlanValidationError) as exc_info:
        validate_engineering_plan(plan, enabled_roles={"backend"}, max_files_per_task=2)

    assert "MAX_FILES_PER_TASK" in str(exc_info.value)


def test_acceptance_criteria_five_line_limit_rejected():
    plan = EngineeringPlan(
        goal="Long criteria",
        tasks=[
            TaskPlanItem(
                id="task-1",
                title="Too many criteria",
                role="backend",
                description="desc",
                acceptance_criteria=["c1", "c2", "c3", "c4", "c5", "c6"],  # 6 lines > max 5
                write_scope=["src/a.py"],
            )
        ],
    )

    with pytest.raises(PlanValidationError) as exc_info:
        validate_engineering_plan(plan, enabled_roles={"backend"})

    assert "exceeding limit of 5 lines" in str(exc_info.value)


def test_parallel_scope_overlap_rejected():
    # task-1 and task-2 have NO dependency between them, but both write to src/calc.py!
    plan = EngineeringPlan(
        goal="Overlapping parallel tasks",
        tasks=[
            TaskPlanItem(
                id="task-1",
                title="Task 1",
                role="backend",
                description="desc",
                acceptance_criteria=["crit"],
                write_scope=["src/calc.py"],
            ),
            TaskPlanItem(
                id="task-2",
                title="Task 2",
                role="backend",
                description="desc",
                acceptance_criteria=["crit"],
                depends_on=[],  # No dependency -> parallel-capable!
                write_scope=["src/calc.py"],
            ),
        ],
    )

    with pytest.raises(PlanValidationError) as exc_info:
        validate_engineering_plan(plan, enabled_roles={"backend"})

    assert "overlapping write scope" in str(exc_info.value)


def test_sequential_scope_overlap_allowed():
    # task-2 depends on task-1 -> sequentially ordered, so write scope overlap is permitted
    plan = EngineeringPlan(
        goal="Sequential overlapping tasks",
        tasks=[
            TaskPlanItem(
                id="task-1",
                title="Task 1",
                role="backend",
                description="desc",
                acceptance_criteria=["crit"],
                write_scope=["src/calc.py"],
            ),
            TaskPlanItem(
                id="task-2",
                title="Task 2",
                role="backend",
                description="desc",
                acceptance_criteria=["crit"],
                depends_on=["task-1"],  # Explicit dependency
                write_scope=["src/calc.py"],
            ),
        ],
    )

    validate_engineering_plan(plan, enabled_roles={"backend"})


def test_repair_prompt_generation():
    prompt = build_plan_repair_prompt('{"goal": "bad"}', ["Cycle detected", "Role 'frontend' disabled"])
    assert "Cycle detected" in prompt
    assert "Role 'frontend' disabled" in prompt
    assert "at most 2 write_scope entries" in prompt


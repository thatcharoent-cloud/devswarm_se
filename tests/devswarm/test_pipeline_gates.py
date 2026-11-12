"""
tests.devswarm.test_pipeline_gates — Unit tests for Phase C hybrid pipeline & human gates.

Verifies:
  - PMSpecification schema validation, uniqueness of AC IDs, and spec.md renderer
  - ArchitectureDesign schema validation, contract syntax checks, and architecture.md renderer
  - QATestSuite schema and RED-check validator (catches syntax errors vs vacuous vs valid red tests)
  - Human Gate 1 (approve-spec) and Human Gate 2 (approve-design) blocking & approval
  - Task brief synthesis and token boundary (strictly under 1,500 tokens)
  - Traceability Matrix (AC -> Task -> Test -> Gate) and --waive-ac logic
"""

from pathlib import Path
import pytest
from pydantic import ValidationError

from devswarm.agents.pm_schema import AcceptanceCriterion, PMSpecification, render_spec_markdown
from devswarm.agents.architect_schema import (
    ADRRecord,
    ArchitectureDesign,
    InterfaceContract,
    render_architecture_markdown,
)
from devswarm.agents.qa_schema import (
    QASyntaxOrImportError,
    QATestFile,
    QATestSuite,
    QAVacuousTestsError,
    RedCheckValidator,
)
from devswarm.pipeline import GatePausedException, PipelineController
from devswarm.platform.events import EventLog
from devswarm.platform.sandbox import MockSandbox
from devswarm.traceability import TraceabilityMatrix


def test_pm_schema_validation_and_renderer():
    # 1. Valid Specification
    spec_data = {
        "problem_statement": "Users need an interactive calculator to perform arithmetic.",
        "user_stories": [
            "As a user, I want to add numbers so that I can compute totals.",
            "As a user, I want to view calculation history so that I can review past steps.",
        ],
        "acceptance_criteria": [
            {
                "id": "AC-1",
                "description": "add(a, b) returns correct arithmetic sum.",
                "verification_method": "Automated unit test with floats and integers",
                "observable_result": "add(2, 3) == 5",
            },
            {
                "id": "AC-2",
                "description": "Calculator tracks history of past operations.",
                "verification_method": "Automated unit test checking history list",
                "observable_result": "get_history() returns operation strings",
            },
        ],
        "non_goals": ["No mobile GUI or web interface"],
        "assumptions": ["Python 3 standard library only"],
        "open_questions": [],
        "milestones": [{"title": "Phase 1", "target": "Core calculation engine"}],
        "risks": ["Division by zero precision handling"],
    }

    spec = PMSpecification.model_validate(spec_data)
    assert len(spec.acceptance_criteria) == 2
    assert spec.acceptance_criteria[0].id == "AC-1"

    md = render_spec_markdown(spec, run_id="run_test_01")
    assert "# Product Feature Specification" in md
    assert "`AC-1`" in md
    assert "`AC-2`" in md
    assert "No mobile GUI" in md

    # 2. Duplicate AC ID rejected
    invalid_data = dict(spec_data)
    invalid_data["acceptance_criteria"] = [
        spec_data["acceptance_criteria"][0],
        spec_data["acceptance_criteria"][0],  # duplicate AC-1
    ]
    with pytest.raises(ValidationError):
        PMSpecification.model_validate(invalid_data)

    # 3. Invalid AC ID format rejected
    invalid_id_data = dict(spec_data)
    invalid_id_data["acceptance_criteria"] = [
        {
            "id": "Criterion_One",
            "description": "Valid description test",
            "verification_method": "Unit test",
            "observable_result": "True",
        }
    ]
    with pytest.raises(ValidationError):
        PMSpecification.model_validate(invalid_id_data)


def test_architect_schema_and_contract_validation():
    valid_contract = InterfaceContract(
        path="src/calculator.py",
        content=(
            "class Calculator:\n"
            "    def add(self, a: float, b: float) -> float:\n"
            "        '''Add two numbers.'''\n"
            "        raise NotImplementedError()\n"
        ),
        description="Core calculator contract",
    )

    design_data = {
        "stack_decision": "Python standard library (CLI)",
        "stack_justification": "Zero external dependencies for maximum portability.",
        "sources_cited": [{"title": "Python Math", "url": "https://docs.python.org/3/library/math.html"}],
        "folder_structure": ["src/", "src/calculator.py", "tests/", "tests/test_calculator.py"],
        "data_models": ["Calculator class with state"],
        "components": ["ArithmeticEngine", "HistoryTracker"],
        "contracts": [valid_contract],
        "adrs": [
            {
                "id": "ADR-001",
                "title": "Use standard library only",
                "status": "Accepted",
                "context": "Local execution without external package overhead",
                "decision": "Use Python stdlib math/argparse",
                "consequences": "Fast, deterministic gates",
                "alternatives_considered": ["NumPy", "SymPy"],
            }
        ],
        "security_considerations": ["Prevent eval/exec injection"],
        "implementer_constraints": ["Max 150 lines per file"],
    }

    design = ArchitectureDesign.model_validate(design_data)
    assert len(design.contracts) == 1
    assert design.contracts[0].path == "src/calculator.py"

    md = render_architecture_markdown(design, run_id="run_test_01")
    assert "# Software Architecture & System Design" in md
    assert "ADR-001" in md
    assert "src/calculator.py" in md

    # Contract with broken syntax rejected
    with pytest.raises(ValidationError) as exc_info:
        InterfaceContract(
            path="src/broken.py",
            content="def broken_syntax(x:\n   return x",  # Missing closing paren
        )
    assert "invalid Python syntax" in str(exc_info.value)


@pytest.mark.asyncio
async def test_qa_red_check_validator(tmp_path):
    mock_sandbox = MockSandbox()

    test_file = QATestFile(
        path="tests/test_calculator.py",
        content=(
            "import pytest\n"
            "from src.calculator import Calculator\n\n"
            "def test_add():\n"
            "    calc = Calculator()\n"
            "    assert calc.add(2, 3) == 5\n"
        ),
    )
    suite = QATestSuite(test_files=[test_file], ac_mapping={"AC-1": ["test_add"]})

    # 1. Collection failure (syntax/import error)
    mock_sandbox.script_command(
        "pytest --collect-only -q",
        exit_code=1,
        stderr="ImportError: No module named 'src.calculator'",
    )
    with pytest.raises(QASyntaxOrImportError):
        await RedCheckValidator.validate_red_check(suite, mock_sandbox, worktree_dir=str(tmp_path))

    # 2. Premature pass (vacuous test)
    mock_sandbox.script_command("pytest --collect-only -q", exit_code=0, stdout="1 test collected")
    mock_sandbox.script_command("pytest -q", exit_code=0, stdout="1 passed in 0.01s")
    with pytest.raises(QAVacuousTestsError):
        await RedCheckValidator.validate_red_check(suite, mock_sandbox, worktree_dir=str(tmp_path))

    # 3. Successful RED condition (collects cleanly, fails on unimplemented stub)
    mock_sandbox.script_command("pytest --collect-only -q", exit_code=0, stdout="1 test collected")
    mock_sandbox.script_command(
        "pytest -q",
        exit_code=1,
        stdout="FAILED tests/test_calculator.py::test_add - NotImplementedError",
    )
    passed, msg = await RedCheckValidator.validate_red_check(suite, mock_sandbox, worktree_dir=str(tmp_path))
    assert passed is True
    assert "RED-check PASSED" in msg

    # 4. Successful RED condition with mixed tracebacks (NotImplementedError + subprocess or inner ImportError)
    mock_sandbox.script_command("pytest --collect-only -q", exit_code=0, stdout="2 tests collected")
    mock_sandbox.script_command(
        "pytest -q",
        exit_code=1,
        stdout=(
            "FAILED tests/test_calculator.py::test_add - NotImplementedError: Contract stub\n"
            "FAILED tests/test_cli.py::test_cli_subp - AssertionError: ModuleNotFoundError: No module named 'calc.cli'"
        ),
    )
    passed, msg = await RedCheckValidator.validate_red_check(suite, mock_sandbox, worktree_dir=str(tmp_path))
    assert passed is True
    assert "RED-check PASSED" in msg


def test_human_gates_and_task_briefs(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    artifacts_dir = tmp_path / ".swarm" / "runs" / "run_001" / "artifacts"
    artifacts_dir.mkdir(parents=True)
    run_dir = artifacts_dir.parent

    event_log = EventLog(run_dir=run_dir, run_id="run_001")
    sandbox = MockSandbox()

    ctrl = PipelineController(
        run_id="run_001",
        repo_path=repo_dir,
        artifacts_dir=artifacts_dir,
        event_log=event_log,
        sandbox=sandbox,
    )

    # Gate 1 check without approval pauses
    with pytest.raises(GatePausedException) as exc_info:
        ctrl.check_gate_1_spec_approval()
    assert "PAUSED AT GATE 1" in str(exc_info.value)

    # Approve Gate 1
    ctrl.approve_spec("Looks good")
    assert ctrl.check_gate_1_spec_approval() is True

    # Gate 2 check without approval pauses
    with pytest.raises(GatePausedException) as exc_info:
        ctrl.check_gate_2_design_approval()
    assert "PAUSED AT GATE 2" in str(exc_info.value)

    # Approve Gate 2
    ctrl.approve_design("Approved contracts")
    assert ctrl.check_gate_2_design_approval() is True

    # Auto-approve design flag
    ctrl_auto = PipelineController(
        run_id="run_002",
        repo_path=repo_dir,
        artifacts_dir=artifacts_dir,
        event_log=event_log,
        sandbox=sandbox,
        auto_approve_design=True,
    )
    assert ctrl_auto.check_gate_2_design_approval() is True

    # Test Task Brief Synthesis
    ctrl.spec = PMSpecification(
        problem_statement="Test calculator",
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-1",
                description="Add returns sum",
                verification_method="pytest",
                observable_result="add(2, 3) == 5",
            )
        ],
    )
    ctrl.design = ArchitectureDesign(
        stack_decision="Python",
        stack_justification="Clean",
        folder_structure=["src/"],
        components=["Calc"],
        contracts=[
            InterfaceContract(path="src/calc.py", content="def add(a, b): raise NotImplementedError()")
        ],
    )

    brief = ctrl.generate_task_brief(
        task_id="task_1",
        task_title="Implement add function",
        task_description="Implement addition in src/calc.py",
        write_scope=["src/calc.py"],
    )

    assert "# Task Brief: [task_1]" in brief
    assert "AC-1" in brief
    assert "def add(a, b):" in brief

    # Task brief must be strictly under 1,500 tokens (approx 6,000 characters)
    assert len(brief) < 6000


def test_traceability_matrix(tmp_path):
    spec = PMSpecification(
        problem_statement="Calculation system",
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-1",
                description="Add two numbers",
                verification_method="pytest",
                observable_result="5",
            ),
            AcceptanceCriterion(
                id="AC-2",
                description="Track calculation history",
                verification_method="pytest",
                observable_result="['2 + 3 = 5']",
            ),
        ],
    )

    planned_tasks = [
        {"id": "task_1", "title": "Implement arithmetic AC-1", "description": "Add AC-1"},
        {"id": "task_2", "title": "Implement history AC-2", "description": "History AC-2"},
    ]

    qa_map = {
        "AC-1": ["tests/test_calc.py::test_add"],
        "AC-2": ["tests/test_calc.py::test_history"],
    }

    matrix = TraceabilityMatrix.create_from_spec(spec, planned_tasks=planned_tasks, qa_mapping=qa_map)
    assert len(matrix.records) == 2
    assert "task_1" in matrix.records["AC-1"].tasks
    assert "task_2" in matrix.records["AC-2"].tasks

    # Initially pending
    ready, unres = matrix.is_review_ready()
    assert ready is False
    assert len(unres) == 2

    # Update with passing test output
    matrix.update_test_results("PASSED tests/test_calc.py::test_add", exit_code=0)
    ready, unres = matrix.is_review_ready()
    assert ready is True

    # Test with waived AC
    matrix_waived = TraceabilityMatrix.create_from_spec(
        spec, planned_tasks=planned_tasks, qa_mapping=qa_map, waived_acs=["AC-2"]
    )
    assert matrix_waived.records["AC-2"].status == "WAIVED"
    ready, unres = matrix_waived.is_review_ready()
    # AC-1 still pending
    assert ready is False
    assert "AC-1" in unres[0]
    # Mark AC-1 passed
    matrix_waived.records["AC-1"].status = "PASS"
    ready, unres = matrix_waived.is_review_ready()
    assert ready is True

    # Markdown format check
    md = matrix.to_markdown()
    assert "Acceptance Criteria Traceability Matrix" in md
    assert "`AC-1`" in md
    assert "`PASS`" in md

"""
tests/devswarm/test_roles.py — Tests for role definitions, scopes, and prompt templates.
"""

from pathlib import Path
import pytest

from devswarm.agents.roles import RoleSpec, RoleRegistry
from devswarm.agents.prompts import build_system_prompt


def test_role_spec_defaults() -> None:
    role = RoleSpec(name="custom")
    assert role.name == "custom"
    assert role.enabled is True
    assert "read_file" in role.allowed_tools
    assert role.max_turns == 12


def test_role_registry_loading() -> None:
    registry = RoleRegistry("config/roles.yaml")
    enabled = registry.get_enabled_roles()

    # Backend and tester must be enabled by default
    assert "backend" in enabled
    assert "tester" in enabled

    # Others disabled by default
    assert "architect" not in enabled
    assert "frontend" not in enabled
    assert "security" not in enabled

    backend_role = registry.get_role("backend")
    assert backend_role.enabled is True
    assert "pytest" in backend_role.command_allowlist
    assert "*.py" in backend_role.default_write_scope or "src/**" in backend_role.default_write_scope


def test_role_registry_nonexistent_role() -> None:
    registry = RoleRegistry("config/roles.yaml")
    with pytest.raises(KeyError):
        registry.get_role("non_existent_role_xyz")


def test_build_system_prompt_backend() -> None:
    prompt = build_system_prompt(
        role="backend",
        task_id="task-1",
        task_title="Add multiply helper",
        task_description="Implement multiply function",
        write_scope=["calc.py"],
        acceptance_criteria=["Unit tests pass", "No syntax errors"],
        max_files=2,
    )

    # Must contain untrusted data warning
    assert "UNTRUSTED DATA" in prompt
    assert "<<<UNTRUSTED_CONTENT>>>" in prompt
    # Must contain write scope and max files constraint
    assert "'calc.py'" in prompt
    assert "2" in prompt
    # Must contain no git command rule
    assert "NO GIT COMMANDS" in prompt
    # Must specify role
    assert "Backend Specialist Engineer" in prompt
    assert "Add multiply helper" in prompt


def test_build_system_prompt_tester() -> None:
    prompt = build_system_prompt(
        role="tester",
        task_id="task-2",
        task_title="Verify multiply helper",
        task_description="Write pytest tests",
        write_scope=["test_calc.py"],
        acceptance_criteria=["100% coverage"],
        max_files=1,
    )

    assert "Quality Assurance & Test Engineer" in prompt
    assert "pytest" in prompt
    assert "'test_calc.py'" in prompt


def test_role_provider_bindings() -> None:
    from devswarm.providers.registry import ProviderRegistry
    reg = ProviderRegistry("config/providers.yaml", "config/roles.yaml", mock_mode=True)

    pm_prov, pm_mod, _, _, _ = reg.get_role_binding("pm")
    assert pm_prov.name == "siliconflow"
    assert "DeepSeek" in pm_mod

    arch_prov, arch_mod, _, _, _ = reg.get_role_binding("architect")
    assert arch_prov.name == "siliconflow"
    assert "DeepSeek" in arch_mod

    qa_prov, qa_mod, _, _, _ = reg.get_role_binding("qa")
    assert qa_prov.name == "siliconflow"
    assert "DeepSeek" in qa_mod

    be_prov, be_mod, _, _, _ = reg.get_role_binding("backend")
    assert be_prov.name in ("ollama", "siliconflow")
    assert any(k in be_mod.lower() for k in ("qwen", "kimi", "deepseek"))

    test_prov, test_mod, _, _, _ = reg.get_role_binding("tester")
    assert test_prov.name in ("ollama", "siliconflow")
    assert any(k in test_mod.lower() for k in ("qwen", "kimi", "deepseek"))

    rev_prov, rev_mod, _, _, _ = reg.get_role_binding("reviewer")
    assert rev_prov.name == "siliconflow"
    assert "DeepSeek" in rev_mod


def test_orchestrator_roles_cli_override(tmp_path) -> None:
    from devswarm.dev_orchestrator import DevOrchestrator
    from devswarm.llm.mock_llm import MockLLM
    from devswarm.platform.sandbox import MockSandbox

    orch_default = DevOrchestrator(
        repo_path=tmp_path,
        goal="Default run",
        client=MockLLM(),
        sandbox=MockSandbox(),
        swarm_dir=tmp_path / "swarm1",
    )
    # Default without --roles: only local coders enabled
    assert orch_default.enabled_roles == {"backend", "tester"}

    orch_custom = DevOrchestrator(
        repo_path=tmp_path,
        goal="Custom run",
        client=MockLLM(),
        sandbox=MockSandbox(),
        swarm_dir=tmp_path / "swarm2",
        roles="pm,architect,qa,reviewer,backend,tester",
    )
    assert orch_custom.enabled_roles == {"pm", "architect", "qa", "reviewer", "backend", "tester"}



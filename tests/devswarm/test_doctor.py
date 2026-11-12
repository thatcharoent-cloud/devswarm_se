"""
tests/devswarm/test_doctor.py — Unit tests for Doctor diagnostics with mocked probes.
"""

import json
import pytest
from pathlib import Path

from devswarm.llm.doctor import Doctor, CheckResult, DoctorReport
from devswarm.llm.mock_llm import MockLLM
from devswarm.llm.ollama_client import OllamaChatResponse


def make_mock_runner(
    ram_bytes: int = 17179869184,  # 16 GB
    docker_available: bool = True,
    is_ac_power: bool = True,
):
    def runner(cmd: list[str]) -> tuple[int, str, str]:
        cmd_str = " ".join(cmd)
        if "hw.memsize" in cmd_str:
            return 0, str(ram_bytes), ""
        if "docker" in cmd_str:
            if docker_available:
                return 0, json.dumps({"MemTotal": 8589934592, "NCPU": 4}), ""
            return 1, "", "docker command not found"
        if "pmset" in cmd_str:
            if is_ac_power:
                return 0, "Now drawing from 'AC Power'\n", ""
            return 0, "Now drawing from 'Battery Power' (75% remaining)\n", ""
        return 0, "", ""
    return runner


@pytest.mark.asyncio
async def test_doctor_all_pass(tmp_path, monkeypatch):
    """Test doctor report when all host and model probes pass."""
    mock_llm = MockLLM(
        version="0.5.12",
        installed_models=["qwen2.5-coder:7b"],
        loaded_models=[{"name": "qwen2.5-coder:7b", "processor": "100% GPU"}],
    )

    # 1. Tiny gen response
    mock_llm.add_response(content="OK", tokens_per_sec=38.0)
    # 2. Tool call smoke test (5 trials, all succeed)
    for _ in range(5):
        mock_llm.add_response(
            content="",
            tool_calls=[{"function": {"name": "add_numbers", "arguments": {"a": 12, "b": 30}}}],
        )
    # 3. Structured output response
    mock_llm.add_response(content=json.dumps({"status": "ready", "code": 200}))
    # 4. Context window probe (marker retrieval)
    mock_llm.add_response(content="The secret code was SWARM_SECRET_KEY_789.")

    doc = Doctor(
        client=mock_llm,
        config_path="config/models.yaml",
        artifact_dir=tmp_path,
        command_runner=make_mock_runner(ram_bytes=17179869184, docker_available=True, is_ac_power=True),
        model="qwen2.5-coder:7b",
    )

    # Mock Darwin platform
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")

    report = await doc.run_diagnostics()

    assert report.overall_status in ("PASS", "WARN")
    # Artifact created
    artifact_file = tmp_path / "doctor_report.md"
    assert artifact_file.exists()
    content = artifact_file.read_text()
    assert "Swarm Doctor Diagnostic Report" in content
    assert "qwen2.5-coder:7b" in content


@pytest.mark.asyncio
async def test_doctor_model_missing_fails_with_pull_command(tmp_path, monkeypatch):
    """Test doctor reports FAIL and outputs exact ollama pull command when model is missing."""
    mock_llm = MockLLM(
        version="0.5.12",
        installed_models=["other-model:latest"],  # Configured model missing
    )

    doc = Doctor(
        client=mock_llm,
        config_path="config/models.yaml",
        artifact_dir=tmp_path,
        command_runner=make_mock_runner(),
        model="qwen2.5-coder:7b",
    )

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")

    report = await doc.run_diagnostics()

    assert report.overall_status == "FAIL"
    model_check = next(c for c in report.checks if c.name == "Configured Model Presence")
    assert model_check.status == "FAIL"
    assert "ollama pull" in model_check.recommendation


@pytest.mark.asyncio
async def test_doctor_battery_warning(tmp_path, monkeypatch):
    """Test doctor warns when running on battery power."""
    mock_llm = MockLLM(
        installed_models=["qwen2.5-coder:7b"],
    )
    mock_llm.add_response(content="OK", tokens_per_sec=30.0)
    for _ in range(5):
        mock_llm.add_response(
            content="",
            tool_calls=[{"function": {"name": "add_numbers", "arguments": {"a": 12, "b": 30}}}],
        )
    mock_llm.add_response(content=json.dumps({"status": "ready", "code": 200}))
    mock_llm.add_response(content="SWARM_SECRET_KEY_789")

    doc = Doctor(
        client=mock_llm,
        config_path="config/models.yaml",
        artifact_dir=tmp_path,
        command_runner=make_mock_runner(is_ac_power=False),  # On battery
        model="qwen2.5-coder:7b",
    )

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")

    report = await doc.run_diagnostics()

    batt_check = next(c for c in report.checks if c.name == "Laptop Power Source")
    assert batt_check.status == "WARN"
    assert "Battery" in batt_check.message


@pytest.mark.asyncio
async def test_doctor_tool_calling_failure(tmp_path, monkeypatch):
    """Test doctor fails when tool calling succeeds in <4 of 5 trials."""
    mock_llm = MockLLM(
        installed_models=["qwen2.5-coder:7b"],
    )
    mock_llm.add_response(content="OK", tokens_per_sec=25.0)
    # Only 2 of 5 tool calls succeed
    for i in range(5):
        if i < 2:
            mock_llm.add_response(
                content="",
                tool_calls=[{"function": {"name": "add_numbers", "arguments": {"a": 12, "b": 30}}}],
            )
        else:
            mock_llm.add_response(content="Sorry, I cannot call tools.")
    mock_llm.add_response(content=json.dumps({"status": "ready", "code": 200}))
    mock_llm.add_response(content="SWARM_SECRET_KEY_789")

    doc = Doctor(
        client=mock_llm,
        config_path="config/models.yaml",
        artifact_dir=tmp_path,
        command_runner=make_mock_runner(),
        model="qwen2.5-coder:7b",
    )

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")

    report = await doc.run_diagnostics()
    tc_check = next(c for c in report.checks if c.name == "Tool-Call Smoke Test")
    assert tc_check.status == "FAIL"
    assert report.overall_status == "FAIL"

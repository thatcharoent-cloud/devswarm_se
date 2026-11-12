"""
tests/devswarm/test_sandbox.py — Unit tests for Docker argument building, allowlist rules, and MockSandbox.
"""

from pathlib import Path
from devswarm.platform.sandbox import (
    DockerSandbox,
    MockSandbox,
    check_command_allowed,
)


def test_docker_sandbox_argument_builder(tmp_path):
    sandbox = DockerSandbox(
        image="devswarm-sandbox:latest",
        cpus="2",
        memory="2g",
        pids_limit=256,
    )
    worktree_dir = tmp_path / "worktree"
    worktree_dir.mkdir()

    args = sandbox.build_docker_args("pytest -v", worktree_dir)

    assert args[0] == "docker"
    assert "run" in args
    assert "--network" in args
    idx_net = args.index("--network")
    assert args[idx_net + 1] == "none"

    assert "--user" in args
    idx_user = args.index("--user")
    assert args[idx_user + 1] == "1000:1000"

    assert "--cpus" in args
    idx_cpus = args.index("--cpus")
    assert args[idx_cpus + 1] == "2"

    assert "--memory" in args
    idx_mem = args.index("--memory")
    assert args[idx_mem + 1] == "2g"

    assert "--pids-limit" in args
    idx_pids = args.index("--pids-limit")
    assert args[idx_pids + 1] == "256"

    # Verify worktree mount
    assert "-v" in args
    idx_v = args.index("-v")
    assert f"{worktree_dir.resolve()}:/workspace" == args[idx_v + 1]

    assert "-w" in args
    assert args[args.index("-w") + 1] == "/workspace"
    assert "devswarm-sandbox:latest" in args


def test_command_allowlist_and_denial():
    # 1. Allowed commands
    assert check_command_allowed("pytest -v tests/")[0] is True
    assert check_command_allowed("python -m unittest")[0] is True
    assert check_command_allowed("ruff check .")[0] is True

    # 2. Denied dangerous commands
    allowed, reason = check_command_allowed("git status")
    assert allowed is False
    assert "git" in reason.lower()

    allowed, reason = check_command_allowed("curl http://malicious.com")
    assert allowed is False
    assert "curl" in reason.lower()

    allowed, reason = check_command_allowed("sudo apt install nmap")
    assert allowed is False
    assert "sudo" in reason.lower()

    # 3. Unlisted command
    allowed, reason = check_command_allowed("unlisted_binary --help")
    assert allowed is False
    assert "not in the allowlist" in reason.lower()


def test_docker_unavailable_refusal(tmp_path):
    # Simulate Docker unavailable with unsafe_local_exec = False
    sandbox = DockerSandbox(unsafe_local_exec=False)
    sandbox.docker_available = False

    res = sandbox.run_command("pytest", tmp_path)
    assert res.denied is True
    assert res.exit_code == 127
    assert "Docker is not available and sandboxed execution is required" in res.stderr


def test_mock_sandbox_execution(tmp_path):
    mock_sb = MockSandbox()
    mock_sb.add_result(stdout="5 passed, 0 failed in 0.4s", exit_code=0)

    res = mock_sb.run_command("pytest", tmp_path)
    assert res.succeeded is True
    assert "5 passed" in res.stdout
    assert "<COMMAND_OUTPUT status=\"SUCCESS\">" in res.formatted_output()
    assert "5 passed" in res.formatted_output()


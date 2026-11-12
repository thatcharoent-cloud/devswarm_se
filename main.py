#!/usr/bin/env python3
"""
main.py — CLI entry-point for the Autonomous Software Engineering Swarm (DevSwarm).

Runs 100% locally on Apple Silicon via Ollama (Metal GPU) and isolated Docker sandboxes.

Commands:
    # Run the swarm on a codebase to achieve a goal:
    python main.py --repo /path/to/repo --goal "Add divide function with unit tests"
    python main.py "Add user login" --repo . --interactive

    # Diagnostic & Benchmark tools:
    python main.py doctor
    python main.py bench-models
    python main.py prepare-image

    # Run inspection, control, and pull request approvals:
    python main.py status
    python main.py stop
    python main.py approve [run_id]
    python main.py reject [run_id] --reason "Needs more tests"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from devswarm.approval import ApprovalManager
from devswarm.dev_orchestrator import DevOrchestrator
from devswarm.llm.bench import ModelBenchmark
from devswarm.llm.doctor import Doctor
from devswarm.llm.mock_llm import MockLLM
from devswarm.llm.ollama_client import OllamaClient
from devswarm.platform.guards import KillSwitch
from devswarm.platform.sandbox import DockerSandbox, MockSandbox

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the clean DevSwarm argument parser."""
    p = argparse.ArgumentParser(
        prog="devswarm",
        description="DevSwarm: Autonomous Local Software Engineering Swarm running on Apple Silicon via Ollama.",
    )

    subparsers = p.add_subparsers(dest="subcommand", help="Available subcommands")

    # Subcommand: doctor
    sub_doc = subparsers.add_parser("doctor", help="Check local environment, Metal GPU offload, and Ollama capabilities.")
    sub_doc.add_argument("--mock", action="store_true", help="Run doctor probe in mock mode.")

    # Subcommand: bench-models
    sub_bench = subparsers.add_parser("bench-models", help="Benchmark candidate models on this hardware sequentially.")
    sub_bench.add_argument("--mock", action="store_true", help="Run benchmark in mock mode.")

    # Subcommand: prepare-image
    sub_prep = subparsers.add_parser("prepare-image", help="Build the Docker sandbox image with preinstalled tools.")
    sub_prep.add_argument("--sandbox-image", default="devswarm-sandbox:latest", help="Target image tag to build.")

    # Subcommand: status
    sub_stat = subparsers.add_parser("status", help="Show status table of recent or specified swarm run.")
    sub_stat.add_argument("run_id_arg", nargs="?", default=None, help="Target run ID (defaults to latest).")

    # Subcommand: stop
    sub_stop = subparsers.add_parser("stop", help="Activate kill switch to halt all running tasks cleanly.")
    sub_stop.add_argument("run_id_arg", nargs="?", default=None, help="Target run ID.")

    # Subcommand: approve
    sub_app = subparsers.add_parser("approve", help="Approve pull request and merge integration branch into main.")
    sub_app.add_argument("run_id_arg", nargs="?", default=None, help="Target run ID to approve.")
    sub_app.add_argument("--run-id", dest="run_id", default=None, help="Alternative flag for run ID.")
    sub_app.add_argument("--repo", default=None, help="Path to repository (default: auto-detected from run manifest).")
    sub_app.add_argument("--force", "-f", action="store_true", default=False, help="Force merge by auto-stashing uncommitted repository changes.")

    # Subcommand: approve-spec (GATE 1)
    sub_app_spec = subparsers.add_parser("approve-spec", help="Approve PM feature specification (Human Gate 1).")
    sub_app_spec.add_argument("run_id_arg", nargs="?", default=None, help="Target run ID to approve spec for.")
    sub_app_spec.add_argument("--run-id", dest="run_id", default=None, help="Alternative flag for run ID.")
    sub_app_spec.add_argument("--notes", default="Human approved specification", help="Approval notes or feedback.")

    # Subcommand: approve-design (GATE 2)
    sub_app_des = subparsers.add_parser("approve-design", help="Approve Software Architecture design and contract stubs (Human Gate 2).")
    sub_app_des.add_argument("run_id_arg", nargs="?", default=None, help="Target run ID to approve design for.")
    sub_app_des.add_argument("--run-id", dest="run_id", default=None, help="Alternative flag for run ID.")
    sub_app_des.add_argument("--notes", default="Human approved architecture design", help="Approval notes or feedback.")

    # Subcommand: reject
    sub_rej = subparsers.add_parser("reject", help="Reject pull request and record reason.")
    sub_rej.add_argument("run_id_arg", nargs="?", default=None, help="Target run ID to reject.")
    sub_rej.add_argument("--run-id", dest="run_id", default=None, help="Alternative flag for run ID.")
    sub_rej.add_argument("--reason", default="Operator rejected pull request", help="Reason for rejection.")
    sub_rej.add_argument("--repo", default=None, help="Path to repository (default: auto-detected from run manifest).")

    # Subcommand: providers (supports "providers doctor")
    sub_prov = subparsers.add_parser("providers", help="Diagnose and check configured LLM providers.")
    sub_prov.add_argument("action", nargs="?", default="doctor", choices=["doctor"], help="Provider action (default: doctor).")
    sub_prov.add_argument("--mock", action="store_true", help="Run provider doctor in mock/dry-run mode without live calls.")

    # Subcommand: cost-estimate
    sub_cost = subparsers.add_parser("cost-estimate", help="Estimate cloud dollar costs for configured models before run.")
    sub_cost.add_argument("--goal", default="Software engineering feature", help="Goal to estimate for.")
    sub_cost.add_argument("goal_pos", nargs="?", default=None, help="Positional goal string.")

    # Core Execution Arguments (when no subcommand is invoked)
    p.add_argument("goal_pos", nargs="?", default=None, help="Software engineering goal (positional alternative to --goal).")
    p.add_argument("--repo", default=".", help="Target repository directory path (default: current directory).")
    p.add_argument("--goal", help="High-level software engineering goal.")
    p.add_argument("--goal-file", help="Path to a markdown or text file containing requirements/goal.")
    p.add_argument("--interactive", action="store_true", default=False, help="Clarify requirements interactively before planning.")
    p.add_argument("--resume", help="Resume a paused or interrupted run by run_id.")
    p.add_argument("--roles", help="Comma-separated roles to enable (default: backend,tester).")
    p.add_argument(
        "--sandbox-image", default=os.getenv("SANDBOX_IMAGE", "devswarm-sandbox:latest"),
        help="Docker image tag used for command execution (default: devswarm-sandbox:latest).",
    )
    p.add_argument(
        "--unsafe-local-exec", action="store_true",
        default=os.getenv("UNSAFE_LOCAL_EXEC", "false").lower() in ("true", "1", "yes"),
        help="Run commands directly on host without Docker container isolation (for local dev/testing only).",
    )
    p.add_argument("--offline", action="store_true", default=False, help="Disable cloud providers and fallback to local Ollama models.")
    p.add_argument("--cloud-sees-code", action="store_true", default=False, help="Allow sending code diffs/source to cloud roles.")
    p.add_argument("--log-payloads", action="store_true", default=False, help="Log full text payloads in outbound audit log.")
    p.add_argument("--auto-approve-design", action="store_true", default=False, help="Automatically approve Gate 2 for low-risk runs.")
    p.add_argument("--waive-ac", action="append", default=[], help="Acceptance Criterion ID to waive (e.g. --waive-ac AC-3).")
    p.add_argument(
        "--escalate-to-cloud", action="store_true", default=False,
        help="Consult cloud debugger after max local fix attempts.",
    )
    p.add_argument("--mock", action="store_true", help="Run with MockLLM and MockSandbox (no real API calls).")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging.")

    return p


async def async_main(args: argparse.Namespace) -> int:
    """Main asynchronous execution router."""
    # 1. Handle Subcommands
    if args.subcommand == "doctor":
        client = MockLLM() if args.mock else OllamaClient()
        doc = Doctor(client=client)
        report = await doc.run_diagnostics()
        print(report.to_markdown())
        print(f"\n📄 Diagnostic report saved to: {report.artifact_path}")
        return 0 if report.overall_status in ("PASS", "WARN") else 1

    if args.subcommand == "bench-models":
        client = MockLLM() if args.mock else OllamaClient()
        bench = ModelBenchmark(client=client)
        print("🚀 Running Swarm Model Benchmark across candidate models...")
        results = await bench.run()
        print(f"\n✅ Completed benchmark for {len(results)} candidate models.")
        print(f"📄 Markdown comparison report saved to: {bench.report_path}")
        report_content = bench.report_path.read_text(encoding="utf-8")
        print("\n" + report_content)
        return 0

    if args.subcommand == "prepare-image":
        print(f"📦 Preparing Docker Sandbox Image: {args.sandbox_image}")
        docker_cli = shutil.which("docker")
        if not docker_cli:
            print("❌ Error: Docker CLI not found. Please install Docker Desktop, Colima, or OrbStack.", file=sys.stderr)
            return 1

        dockerfile = Path("Dockerfile")
        if not dockerfile.exists():
            print("ℹ️ No Dockerfile found in current directory. Creating standard devswarm sandbox Dockerfile...")
            dockerfile.write_text(
                "FROM python:3.11-slim\n"
                "RUN apt-get update && apt-get install -y --no-install-recommends git curl make gcc && rm -rf /var/lib/apt/lists/*\n"
                "RUN pip install --no-cache-dir pytest ruff mypy black\n"
                "RUN useradd -m -u 1000 swarmuser\n"
                "USER swarmuser\n"
                "WORKDIR /workspace\n"
            )

        print(f"🔨 Running: docker build -t {args.sandbox_image} .")
        cmd = ["docker", "build", "-t", args.sandbox_image, "."]
        res = subprocess.run(cmd, check=False)
        if res.returncode == 0:
            print(f"✅ Sandbox image '{args.sandbox_image}' built successfully with network access.")
            return 0
        else:
            print(f"❌ Failed building image (exit code {res.returncode})", file=sys.stderr)
            return res.returncode

    if args.subcommand == "status":
        swarm_runs = Path(".swarm/runs")
        target_id = args.run_id_arg
        if not target_id and swarm_runs.exists():
            runs = sorted([d.name for d in swarm_runs.iterdir() if d.is_dir()])
            target_id = runs[-1] if runs else None

        if not target_id or not (swarm_runs / target_id).exists():
            print("❌ No swarm runs found to report status for.", file=sys.stderr)
            return 1

        orch = DevOrchestrator(repo_path=".", goal="", resume_run_id=target_id)
        orch.print_status_table()
        return 0

    if args.subcommand == "stop":
        kill = KillSwitch(stop_file=".swarm/STOP")
        kill.trigger(reason=f"Operator requested stop via CLI for run {args.run_id_arg or 'current'}")
        if args.run_id_arg:
            run_stop = Path(f".swarm/runs/{args.run_id_arg}/STOP")
            run_stop.parent.mkdir(parents=True, exist_ok=True)
            run_stop.write_text("Stopped via CLI", encoding="utf-8")
        print("🛑 Swarm kill switch activated (.swarm/STOP). Running tasks will halt cleanly at next step.")
        return 0

    if args.subcommand == "approve":
        target_id = args.run_id or args.run_id_arg
        repo_dir = getattr(args, "repo", None)
        force = getattr(args, "force", False)
        mgr = ApprovalManager(repo_path=repo_dir)
        try:
            mgr.approve_run(run_id=target_id, force=force)
            return 0
        except Exception as exc:
            print(f"❌ Approval failed: {exc}", file=sys.stderr)
            return 1

    if args.subcommand == "reject":
        target_id = args.run_id or args.run_id_arg
        repo_dir = getattr(args, "repo", None)
        reason = getattr(args, "reason", "Operator rejected pull request")
        mgr = ApprovalManager(repo_path=repo_dir)
        try:
            mgr.reject_run(run_id=target_id, reason=reason)
            return 0
        except Exception as exc:
            print(f"❌ Rejection failed: {exc}", file=sys.stderr)
            return 1

    if args.subcommand == "providers":
        from devswarm.providers.doctor import ProvidersDoctor
        from devswarm.providers.registry import ProviderRegistry
        is_mock = bool(getattr(args, "mock", False) or "--mock" in sys.argv)
        reg = ProviderRegistry(mock_mode=is_mock)
        doc = ProvidersDoctor(registry=reg)
        report_md = await doc.run_diagnostics(skip_live_calls=is_mock)
        print(report_md)
        return 0

    if args.subcommand == "cost-estimate":
        from devswarm.providers.cost_estimator import CostEstimator
        goal_text = args.goal_pos or args.goal or "Software engineering feature"
        estimator = CostEstimator()
        estimates = estimator.estimate_run(goal_text)
        print(estimator.format_markdown(goal_text, estimates))
        return 0

    if args.subcommand in ("approve-spec", "approve-design"):
        import json
        from datetime import datetime, timezone
        swarm_runs = Path(".swarm/runs")
        target_id = getattr(args, "run_id", None) or getattr(args, "run_id_arg", None)
        if not target_id and swarm_runs.exists():
            runs = sorted([d.name for d in swarm_runs.iterdir() if d.is_dir()])
            target_id = runs[-1] if runs else None

        if not target_id or not (swarm_runs / target_id).exists():
            print(f"❌ Error: Swarm run '{target_id or ''}' not found.", file=sys.stderr)
            return 1

        run_dir = swarm_runs / target_id
        notes = getattr(args, "notes", "Approved via CLI")

        if args.subcommand == "approve-spec":
            marker = run_dir / "SPEC_APPROVED"
            marker.write_text(json.dumps({"approved_at": datetime.now(timezone.utc).isoformat(), "run_id": target_id, "notes": notes}, indent=2), encoding="utf-8")
            print(f"✅ Specification approved for run {target_id}. Resume pipeline with: python main.py --resume {target_id}")
            return 0
        else:
            marker = run_dir / "DESIGN_APPROVED"
            marker.write_text(json.dumps({"approved_at": datetime.now(timezone.utc).isoformat(), "run_id": target_id, "notes": notes}, indent=2), encoding="utf-8")
            print(f"✅ Architecture design approved for run {target_id}. Resume pipeline with: python main.py --resume {target_id}")
            return 0

    # 2. Main Software Engineering Swarm Execution
    goal = args.goal or args.goal_pos
    if not goal and args.goal_file:
        g_path = Path(args.goal_file)
        if g_path.exists():
            goal = g_path.read_text(encoding="utf-8").strip()

    if not goal and not args.resume:
        print("❌ Error: A development goal (e.g. --goal \"...\" or file) is required.", file=sys.stderr)
        print("   Usage: python main.py --repo <path> --goal \"<description>\"", file=sys.stderr)
        return 1

    print("🚀 DevSwarm starting (Local Software Engineering Swarm)")
    print(f"    Repository: {args.repo}")
    print(f"    Goal      : {goal or f'(Resuming run: {args.resume})'}")
    print(f"    Profile   : 16 GB Apple Silicon (macOS Metal GPU)")

    client = MockLLM() if args.mock else None
    sandbox = MockSandbox() if args.mock else DockerSandbox(
        image=args.sandbox_image,
        unsafe_local_exec=args.unsafe_local_exec,
    )

    orch = DevOrchestrator(
        repo_path=args.repo,
        goal=goal or "",
        client=client,
        sandbox=sandbox,
        interactive=args.interactive,
        resume_run_id=args.resume,
        question_prompter=input if args.interactive else None,
        escalate_to_cloud=getattr(args, "escalate_to_cloud", False),
        auto_approve_design=getattr(args, "auto_approve_design", False),
        waive_ac=getattr(args, "waive_ac", []),
        offline=getattr(args, "offline", False),
        cloud_sees_code=getattr(args, "cloud_sees_code", False),
        log_payloads=getattr(args, "log_payloads", False),
        mock_mode=args.mock,
        roles=getattr(args, "roles", None),
    )

    return await orch.execute_run()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    try:
        sys.exit(asyncio.run(async_main(args)))
    except KeyboardInterrupt:
        print("\n🛑 Swarm interrupted by user (Ctrl-C). State preserved.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()

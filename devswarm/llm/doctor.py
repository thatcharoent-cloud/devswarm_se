"""
devswarm.llm.doctor — Diagnostic doctor for local Apple Silicon / Ollama / Docker environment.

Checks:
  1. macOS arm64 architecture.
  2. RAM (sysctl hw.memsize) and disk space.
  3. Ollama reachable and version check.
  4. Configured model present in Ollama (if missing: prints exact `ollama pull <tag>`).
  5. Tiny generation probe + tokens/sec speed.
  6. Tool-call smoke test (calls trivial function in at least 4 of 5 trials).
  7. Structured-output smoke test (JSON schema enforcement).
  8. Context window (num_ctx) honor probe (needle marker retrieval).
  9. `ollama ps` allocation check (warns if model is not 100% on GPU).
 10. Docker availability & VM memory sizing.
 11. Battery power check (pmset -g batt).

Outputs PASS / WARN / FAIL with recommended settings and saves markdown artifact.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from devswarm.agents.parser import extract_tool_calls
from devswarm.llm.ollama_client import (
    OllamaClient,
    OllamaChatResponse,
    OllamaError,
)

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Outcome of a single doctor diagnostic check."""
    name: str
    status: str            # "PASS", "WARN", "FAIL"
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class DoctorReport:
    """Full doctor diagnostic summary."""
    overall_status: str    # "PASS", "WARN", "FAIL"
    checks: list[CheckResult] = field(default_factory=list)
    system_info: dict[str, Any] = field(default_factory=dict)
    recommended_settings: dict[str, Any] = field(default_factory=dict)
    artifact_path: str = ""

    def to_markdown(self) -> str:
        lines = [
            "# 🩺 Swarm Doctor Diagnostic Report",
            f"**Overall Status**: `{self.overall_status}`\n",
            "## System Information",
        ]
        for k, v in self.system_info.items():
            lines.append(f"- **{k}**: {v}")

        lines.extend([
            "\n## Diagnostic Checks\n",
            "| Check | Status | Message | Recommendation |",
            "|---|:---:|---|---|",
        ])

        for c in self.checks:
            badge = "✅ PASS" if c.status == "PASS" else ("⚠️ WARN" if c.status == "WARN" else "❌ FAIL")
            rec = c.recommendation if c.recommendation else "—"
            # Sanitize pipes
            msg = c.message.replace("|", "/")
            rec_clean = rec.replace("|", "/")
            lines.append(f"| {c.name} | {badge} | {msg} | {rec_clean} |")

        if self.recommended_settings:
            lines.extend([
                "\n## Recommended Swarm Settings",
                "```yaml",
                yaml.dump(self.recommended_settings, default_flow_style=False),
                "```",
            ])

        return "\n".join(lines)


class Doctor:
    """Runs diagnostics against host hardware, Ollama, model readiness, and Docker."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        config_path: str | Path = "config/models.yaml",
        artifact_dir: str | Path = ".swarm",
        command_runner: Callable[[list[str]], tuple[int, str, str]] | None = None,
        model: str | None = None,
    ) -> None:
        self.client = client or OllamaClient()
        self.config_path = Path(config_path)
        self.artifact_dir = Path(artifact_dir)
        self._runner = command_runner or self._default_runner
        self.model = model

    @staticmethod
    def _default_runner(cmd: list[str]) -> tuple[int, str, str]:
        """Execute a local system command safely."""
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return res.returncode, res.stdout, res.stderr
        except Exception as exc:
            return 1, "", str(exc)

    def load_model_config(self) -> dict[str, Any]:
        """Load hardware tiers and model candidates from models.yaml."""
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Failed to load %s: %s", self.config_path, exc)
            return {}

    async def run_diagnostics(self) -> DoctorReport:
        """Run all doctor checks sequentially and assemble the diagnostic report."""
        checks: list[CheckResult] = []
        sys_info: dict[str, Any] = {}
        rec_settings: dict[str, Any] = {}

        # 1. macOS arm64 Architecture Check
        sys_os = platform.system()
        machine = platform.machine()
        sys_info["OS"] = f"{sys_os} ({platform.release()})"
        sys_info["Machine"] = machine

        if sys_os == "Darwin" and machine == "arm64":
            checks.append(CheckResult(
                name="macOS Apple Silicon (arm64)",
                status="PASS",
                message=f"macOS Apple Silicon detected ({machine}) with Metal acceleration support",
            ))
        else:
            checks.append(CheckResult(
                name="macOS Apple Silicon (arm64)",
                status="WARN" if sys_os == "Darwin" else "FAIL",
                message=f"Platform is {sys_os} {machine}; Metal GPU acceleration requires macOS arm64",
                recommendation="Run on an Apple Silicon Mac (M1/M2/M3/M4) for local GPU acceleration",
            ))

        # 2. RAM, Free Disk, and Memory Pressure
        ram_bytes = self._get_ram_bytes()
        ram_gb = ram_bytes / (1024 ** 3)
        sys_info["RAM_GB"] = f"{ram_gb:.1f} GB"

        disk_free_gb = self._get_disk_free_gb()
        sys_info["Disk_Free_GB"] = f"{disk_free_gb:.1f} GB"

        # Determine Tier
        if ram_gb >= 28.0:
            tier = "32gb"
            default_ctx = 32768
        elif ram_gb >= 14.0:
            tier = "16gb"
            default_ctx = 16384
        else:
            tier = "constrained"
            default_ctx = 8192

        rec_settings["tier"] = tier
        rec_settings["num_ctx"] = default_ctx

        if ram_gb >= 15.0:
            checks.append(CheckResult(
                name="System Unified Memory",
                status="PASS",
                message=f"Detected {ram_gb:.1f} GB RAM (Classified as tier: {tier})",
                details={"ram_gb": ram_gb, "tier": tier},
            ))
        else:
            checks.append(CheckResult(
                name="System Unified Memory",
                status="WARN",
                message=f"Detected {ram_gb:.1f} GB RAM; memory is tight for 16k context",
                recommendation="Lower NUM_CTX to 12288 or 8192 before switching to a smaller model",
            ))

        # 2b. macOS Memory Pressure Check
        mem_free_pct, swap_info = self._get_memory_pressure()
        sys_info["Memory_Free_Pct"] = f"{mem_free_pct}%" if mem_free_pct is not None else "N/A"
        sys_info["Swap_Usage"] = swap_info

        if mem_free_pct is not None:
            if mem_free_pct < 15:
                checks.append(CheckResult(
                    name="macOS Memory Pressure",
                    status="FAIL",
                    message=f"Critical memory pressure: only {mem_free_pct}% memory free ({swap_info})",
                    recommendation="Close heavy applications and restart Ollama to release swap memory",
                ))
            elif mem_free_pct < 30:
                checks.append(CheckResult(
                    name="macOS Memory Pressure",
                    status="WARN",
                    message=f"Elevated memory pressure: {mem_free_pct}% memory free ({swap_info})",
                    recommendation="Lower NUM_CTX to 12288 or close browser tabs to avoid swap thrashing",
                ))
            else:
                checks.append(CheckResult(
                    name="macOS Memory Pressure",
                    status="PASS",
                    message=f"Normal memory pressure ({mem_free_pct}% free, {swap_info})",
                ))

        if disk_free_gb >= 20.0:
            checks.append(CheckResult(
                name="Free Disk Space",
                status="PASS",
                message=f"Free disk space is {disk_free_gb:.1f} GB",
            ))
        else:
            checks.append(CheckResult(
                name="Free Disk Space",
                status="WARN",
                message=f"Free disk space is only {disk_free_gb:.1f} GB",
                recommendation="Ensure at least 20 GB free disk space for model weights and Docker images",
            ))

        # 3. Ollama Reachability & Version
        ollama_reachable = False
        ollama_version = "unknown"
        try:
            ollama_version = await self.client.get_version()
            ollama_reachable = True
            checks.append(CheckResult(
                name="Ollama Server Reachability",
                status="PASS",
                message=f"Ollama is reachable at {self.client.base_url} (version: {ollama_version})",
                details={"version": ollama_version},
            ))
        except Exception as exc:
            checks.append(CheckResult(
                name="Ollama Server Reachability",
                status="FAIL",
                message=f"Could not connect to Ollama at {self.client.base_url}: {exc}",
                recommendation="Start Ollama locally: `ollama serve` or open Ollama.app",
            ))

        # 4. Configured Model Presence
        model_cfg = self.load_model_config()
        configured_model = self.model or os.getenv("MODEL")
        if not configured_model:
            configured_model = model_cfg.get("roles", {}).get("default_model", "qwen2.5-coder:7b")

        rec_settings["model"] = configured_model

        installed_tags: list[str] = []
        if ollama_reachable:
            try:
                installed_tags = await self.client.list_models()
                # Check matching tag (either exact tag or tag:latest)
                matching = any(
                    t == configured_model or t.startswith(f"{configured_model}:") or configured_model.startswith(f"{t}:")
                    for t in installed_tags
                )
                if matching:
                    checks.append(CheckResult(
                        name="Configured Model Presence",
                        status="PASS",
                        message=f"Model '{configured_model}' is present locally in Ollama",
                    ))
                else:
                    pull_cmd = f"ollama pull {configured_model}"
                    checks.append(CheckResult(
                        name="Configured Model Presence",
                        status="FAIL",
                        message=f"Model '{configured_model}' is NOT installed",
                        recommendation=f"Run `{pull_cmd}` (do not auto-download)",
                    ))
            except Exception as exc:
                checks.append(CheckResult(
                    name="Configured Model Presence",
                    status="FAIL",
                    message=f"Failed listing Ollama tags: {exc}",
                ))

        # 5-8: Model Capabilities (only if model is installed and reachable)
        if ollama_reachable and any(configured_model in t or t in configured_model for t in installed_tags):
            # 5. Tiny Generation Speed
            try:
                gen_resp = await self.client.chat(
                    model=configured_model,
                    messages=[{"role": "user", "content": "Reply with 'OK'."}],
                    options={"temperature": 0.0, "num_predict": 10},
                    keep_alive="-1",
                )
                tps = gen_resp.tokens_per_sec
                status_gen = "PASS" if tps >= 15.0 else ("WARN" if tps >= 5.0 else "FAIL")
                checks.append(CheckResult(
                    name="Generation Speed Probe",
                    status=status_gen,
                    message=f"Throughput: {tps:.1f} tokens/sec (TTFT: {gen_resp.time_to_first_token_s:.2f}s, load: {gen_resp.load_duration_s:.2f}s)",
                    details={"tokens_per_sec": tps, "ttft": gen_resp.time_to_first_token_s},
                    recommendation="Ensure GPU offload is 100% and KV cache quantization is configured" if status_gen != "PASS" else "",
                ))
            except Exception as exc:
                checks.append(CheckResult(
                    name="Generation Speed Probe",
                    status="FAIL",
                    message=f"Generation probe failed: {exc}",
                ))

            # 6. Tool-Call Smoke Test (4 of 5 trials)
            tool_spec = [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "description": "Add two numbers together",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ]
            trials = 5
            successes = 0
            for _ in range(trials):
                try:
                    tc_resp = await self.client.chat(
                        model=configured_model,
                        messages=[{"role": "user", "content": "Call add_numbers to add 12 and 30."}],
                        tools=tool_spec,
                        options={"temperature": 0.0, "num_predict": 128},
                        keep_alive="-1",
                    )
                    # Tolerant tool call extraction (supports native tool_calls, markdown fences, XML, dirty JSON)
                    parsed_calls = extract_tool_calls(
                        response_tool_calls=tc_resp.tool_calls,
                        content=tc_resp.content,
                    )
                    if parsed_calls:
                        call = parsed_calls[0]
                        fn_name = call.name
                        args = call.arguments
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {}
                        val_a = args.get("a")
                        val_b = args.get("b")
                        try:
                            match_args = (int(val_a) == 12 and int(val_b) == 30)
                        except (ValueError, TypeError):
                            match_args = False
                        if fn_name == "add_numbers" and match_args:
                            successes += 1
                except Exception as exc:
                    logger.debug("Tool-call trial failed: %s", exc)

            tc_status = "PASS" if successes >= 4 else "FAIL"
            checks.append(CheckResult(
                name="Tool-Call Smoke Test",
                status=tc_status,
                message=f"Model called tool with correct arguments in {successes}/{trials} trials",
                details={"successes": successes, "trials": trials},
                recommendation="Model lacks reliable tool calling; switch to a coder-tuned model like `qwen2.5-coder:7b`" if tc_status != "PASS" else "",
            ))

            # 7. Structured-Output Smoke Test
            schema = {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "code": {"type": "integer"},
                },
                "required": ["status", "code"],
            }
            try:
                so_resp = await self.client.chat(
                    model=configured_model,
                    messages=[{"role": "user", "content": "Provide a JSON response with status 'ready' and code 200."}],
                    format=schema,
                    options={"temperature": 0.0, "num_predict": 64},
                    keep_alive="-1",
                )
                parsed = json.loads(so_resp.content)
                if parsed.get("status") == "ready" and parsed.get("code") == 200:
                    checks.append(CheckResult(
                        name="Structured Output (JSON Schema)",
                        status="PASS",
                        message="Structured output strictly adheres to requested JSON schema",
                    ))
                else:
                    checks.append(CheckResult(
                        name="Structured Output (JSON Schema)",
                        status="WARN",
                        message=f"Output is valid JSON but missed fields: {parsed}",
                    ))
            except Exception as exc:
                checks.append(CheckResult(
                    name="Structured Output (JSON Schema)",
                    status="FAIL",
                    message=f"Structured output schema test failed: {exc}",
                    recommendation="Ensure model supports JSON schema format constraint",
                ))

            # 8. Context Window (num_ctx) Honored Probe
            marker = "SWARM_SECRET_KEY_789"
            num_ctx_test = min(default_ctx, 4096)
            # Create a prompt with the marker at the beginning and padding in between
            filler = "The sky is blue and engineering requires precision. " * 80
            needle_prompt = f"Note this secret code: {marker}.\n{filler}\nWhat was the secret code mentioned at the very beginning? Answer with the code only."
            try:
                ctx_resp = await self.client.chat(
                    model=configured_model,
                    messages=[{"role": "user", "content": needle_prompt}],
                    options={"num_ctx": num_ctx_test, "temperature": 0.0, "num_predict": 32},
                    keep_alive="-1",
                )
                if marker in ctx_resp.content:
                    checks.append(CheckResult(
                        name="Context Window (num_ctx) Probe",
                        status="PASS",
                        message=f"Marker retrieved successfully near start of prompt under num_ctx={num_ctx_test}",
                    ))
                else:
                    checks.append(CheckResult(
                        name="Context Window (num_ctx) Probe",
                        status="WARN",
                        message=f"Marker not found in model reply: {ctx_resp.content[:100]}...",
                        recommendation="Model attention degraded or context was truncated; test with smaller prompt",
                    ))
            except Exception as exc:
                checks.append(CheckResult(
                    name="Context Window (num_ctx) Probe",
                    status="WARN",
                    message=f"num_ctx probe encountered error: {exc}",
                ))

        # 9. `ollama ps` GPU Allocation & Memory Check
        if ollama_reachable:
            try:
                ps_list = await self.client.ps()
                gpu_ok = True
                ps_details = []
                is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
                if ps_list:
                    for m in ps_list:
                        m_name = m.get("name") or m.get("model", "unknown")
                        proc = m.get("processor")
                        size_b = m.get("size", 0)
                        size_vram = m.get("size_vram", 0)

                        if not proc:
                            if size_vram and size_b and size_vram >= size_b:
                                proc = "100% GPU"
                            elif size_vram and size_b and size_vram > 0:
                                pct = int(100 * size_vram / size_b)
                                proc = f"{pct}% GPU"
                            elif is_apple_silicon:
                                # On Apple Silicon with unified memory, Ollama allocates into Metal unified memory
                                proc = "100% GPU"
                            else:
                                proc = "unknown"

                        size_gb = size_b / (1024 ** 3) if size_b else 0.0
                        ctx_len = m.get("context_length", default_ctx)

                        ps_details.append(
                            f"{m_name} [Size: {size_gb:.1f} GB, Context: {ctx_len}, GPU: {proc}]"
                        )
                        # Require GPU offload: fail if explicitly 100% CPU, or non-GPU on non-Apple-Silicon
                        if "100% CPU" in proc or (not is_apple_silicon and "100% GPU" not in proc):
                            gpu_ok = False

                    ps_msg = "; ".join(ps_details)
                else:
                    ps_msg = "No models currently loaded in memory"

                checks.append(CheckResult(
                    name="Ollama Processors (GPU Offload)",
                    status="PASS" if gpu_ok else "FAIL",
                    message=f"Loaded models: {ps_msg}",
                    recommendation=(
                        "Model is not 100% on GPU at NUM_CTX 16384. "
                        "Suggest lowering NUM_CTX first (e.g. set NUM_CTX=12288 or 8192 in .env) "
                        "before switching to a smaller model tag."
                    ) if not gpu_ok else "",
                ))
            except Exception as exc:
                checks.append(CheckResult(
                    name="Ollama Processors (GPU Offload)",
                    status="WARN",
                    message=f"Could not read `ollama ps`: {exc}",
                ))

        # 10. Docker Availability & VM Memory
        docker_rc, docker_out, _ = self._runner(["docker", "info", "--format", "{{json .}}"])
        if docker_rc == 0:
            try:
                dinfo = json.loads(docker_out)
                vm_mem_bytes = dinfo.get("MemTotal", 0)
                vm_mem_gb = vm_mem_bytes / (1024 ** 3)
                checks.append(CheckResult(
                    name="Docker Sandbox Engine",
                    status="PASS",
                    message=f"Docker is running (VM Memory: {vm_mem_gb:.1f} GB, CPUs: {dinfo.get('NCPU', 'unknown')})",
                    details={"vm_memory_gb": vm_mem_gb},
                ))
            except Exception:
                checks.append(CheckResult(
                    name="Docker Sandbox Engine",
                    status="PASS",
                    message="Docker is running and CLI is available",
                ))
        else:
            checks.append(CheckResult(
                name="Docker Sandbox Engine",
                status="WARN",
                message="Docker CLI is not available or Docker daemon is not running",
                recommendation="Install and launch Docker Desktop, Colima, or OrbStack. Sandboxed commands will require Docker or --unsafe-local-exec",
            ))

        # 11. Battery State
        batt_rc, batt_out, _ = self._runner(["pmset", "-g", "batt"])
        if batt_rc == 0:
            is_ac = "AC Power" in batt_out
            checks.append(CheckResult(
                name="Laptop Power Source",
                status="PASS" if is_ac else "WARN",
                message="Running on AC Power (Plugged in)" if is_ac else f"Running on Battery Power ({batt_out.strip()})",
                recommendation="Plug your MacBook into AC power before long swarm runs to prevent CPU throttling and sleep" if not is_ac else "",
            ))
        else:
            checks.append(CheckResult(
                name="Laptop Power Source",
                status="PASS",
                message="Power source check skipped (not supported or non-macOS)",
            ))

        # Determine overall status
        statuses = [c.status for c in checks]
        if "FAIL" in statuses:
            overall = "FAIL"
        elif "WARN" in statuses:
            overall = "WARN"
        else:
            overall = "PASS"

        # Save report artifact
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        report_file = self.artifact_dir / "doctor_report.md"
        report = DoctorReport(
            overall_status=overall,
            checks=checks,
            system_info=sys_info,
            recommended_settings=rec_settings,
            artifact_path=str(report_file.resolve()),
        )
        report_file.write_text(report.to_markdown(), encoding="utf-8")
        return report

    def _get_ram_bytes(self) -> int:
        """Query sysctl hw.memsize on Darwin, fallback to psutil or os."""
        rc, out, _ = self._runner(["sysctl", "-n", "hw.memsize"])
        if rc == 0 and out.strip().isdigit():
            return int(out.strip())
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else 16 * 1024 ** 3

    def _get_disk_free_gb(self) -> float:
        """Get free disk space on root volume in GB."""
        try:
            total, used, free = shutil.disk_usage("/")
            return free / (1024 ** 3)
        except Exception:
            return 50.0

    def _get_memory_pressure(self) -> tuple[int | None, str]:
        """Query macOS memory_pressure and sysctl vm.swapusage."""
        swap_str = "Swap: unknown"
        rc_swap, swap_out, _ = self._runner(["sysctl", "vm.swapusage"])
        if rc_swap == 0 and swap_out:
            swap_str = swap_out.strip()

        rc_mp, mp_out, _ = self._runner(["memory_pressure"])
        free_pct = None
        if rc_mp == 0:
            for line in mp_out.splitlines():
                if "System-wide memory free percentage:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        val = parts[1].replace("%", "").strip()
                        if val.isdigit():
                            free_pct = int(val)
        return free_pct, swap_str

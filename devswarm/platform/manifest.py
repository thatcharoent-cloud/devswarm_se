"""
devswarm.platform.manifest — Reproducible run manifest for audit and traceability.

Saved at run initialization: .swarm/runs/<run_id>/manifest.json
Contains:
  - run_id and ISO UTC timestamp.
  - sanitized configuration snapshot (zero secrets/API keys).
  - model tag and digest (if available).
  - NUM_CTX, thresholds, budgets.
  - prompt hashes for system prompts.
  - target repo git commit hash (HEAD).
  - tool versions (Python, git, Ollama, Docker).
  - host hardware summary (RAM, chip, architecture, OS).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devswarm.platform.events import compute_prompt_hash, redact_secrets


@dataclass
class RunManifest:
    """Immutable snapshot of the run's environment, configuration, and inputs."""
    run_id: str
    goal: str
    target_repo: str
    target_commit: str
    model: str
    model_digest: str
    num_ctx: int
    prompt_hashes: dict[str, str] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    hardware_summary: dict[str, Any] = field(default_factory=dict)
    tool_versions: dict[str, str] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "goal": self.goal,
            "target_repo": self.target_repo,
            "target_commit": self.target_commit,
            "model": self.model,
            "model_digest": self.model_digest,
            "num_ctx": self.num_ctx,
            "prompt_hashes": self.prompt_hashes,
            "thresholds": self.thresholds,
            "config_snapshot": redact_secrets(self.config_snapshot),
            "hardware_summary": self.hardware_summary,
            "tool_versions": self.tool_versions,
        }

    def save(self, run_dir: Path | str) -> Path:
        """Persist manifest to manifest.json in run_dir."""
        p = Path(run_dir) / "manifest.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p


def build_manifest(
    run_id: str,
    goal: str,
    repo_path: Path | str,
    model: str,
    num_ctx: int = 16384,
    config_snapshot: dict[str, Any] | None = None,
    prompt_dict: dict[str, str] | None = None,
    thresholds: dict[str, Any] | None = None,
    model_digest: str = "",
) -> RunManifest:
    """Collect system telemetry and assemble the RunManifest."""
    repo = Path(repo_path).resolve()

    # 1. Target commit
    commit = "unknown"
    try:
        res = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0:
            commit = res.stdout.strip()
    except Exception:
        pass

    # 2. Hardware summary
    ram_gb = 16.0
    try:
        res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip().isdigit():
            ram_gb = round(int(res.stdout.strip()) / (1024 ** 3), 1)
    except Exception:
        pass

    chip_brand = "Apple Silicon"
    try:
        res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            chip_brand = res.stdout.strip()
    except Exception:
        pass

    hw_info = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "chip": chip_brand,
        "ram_gb": ram_gb,
    }

    # 3. Tool versions
    tools: dict[str, str] = {
        "python": sys.version.split()[0],
    }
    for tool_cmd in ["git", "docker", "ollama"]:
        try:
            r = subprocess.run([tool_cmd, "--version"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                tools[tool_cmd] = r.stdout.strip().split("\n")[0]
            else:
                tools[tool_cmd] = "unavailable"
        except Exception:
            tools[tool_cmd] = "unavailable"

    # 4. Prompt hashes
    hashes: dict[str, str] = {}
    if prompt_dict:
        for name, text in prompt_dict.items():
            hashes[name] = compute_prompt_hash(text)

    # 5. Sanitize config snapshot (ensure no API keys or secrets leak)
    clean_config = redact_secrets(config_snapshot or {})

    return RunManifest(
        run_id=run_id,
        goal=goal,
        target_repo=str(repo),
        target_commit=commit,
        model=model,
        model_digest=model_digest,
        num_ctx=num_ctx,
        prompt_hashes=hashes,
        thresholds=thresholds or {},
        config_snapshot=clean_config,
        hardware_summary=hw_info,
        tool_versions=tools,
    )


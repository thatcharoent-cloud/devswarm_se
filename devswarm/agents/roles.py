"""
devswarm.agents.roles — Specialist role definitions and configuration loader.

Loads and enforces role specifications from config/roles.yaml:
  - Enabled status (default: backend and tester enabled; architect, frontend, security, devops disabled)
  - Tool allowlists per role
  - Default write scopes (narrow globs)
  - Sandbox command allowlists
  - Turn and budget constraints
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

logger = logging.getLogger(__name__)

DEFAULT_ROLES_PATH = Path("config/roles.yaml")


@dataclass
class RoleSpec:
    """Specification for a specialist engineering role."""
    name: str
    enabled: bool = True
    description: str = ""
    provider: str = "ollama"
    model: str = ""
    thinking: bool = False
    max_tokens: int = 2048
    budget_usd: float = 0.0
    search_enabled: bool = False
    allowed_content_classes: list[str] = field(default_factory=lambda: ["*"])
    allowed_tools: list[str] = field(default_factory=lambda: [
        "read_file", "list_dir", "search", "write_file", "edit_file", "run_command"
    ])
    default_write_scope: list[str] = field(default_factory=list)
    command_allowlist: list[str] = field(default_factory=list)
    max_turns: int = 12
    max_tool_calls: int = 20
    token_budget: int = 25000
    minutes_budget: float = 10.0

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> RoleSpec:
        return cls(
            name=name,
            enabled=bool(data.get("enabled", True)),
            description=data.get("description", ""),
            provider=data.get("provider", "ollama"),
            model=data.get("model", ""),
            thinking=bool(data.get("thinking", False)),
            max_tokens=int(data.get("max_tokens", 2048)),
            budget_usd=float(data.get("budget_usd", 0.0)),
            search_enabled=bool(data.get("search_enabled", False)),
            allowed_content_classes=list(data.get("allowed_content_classes", ["*"])),
            allowed_tools=list(data.get("allowed_tools", [
                "read_file", "list_dir", "search", "write_file", "edit_file", "run_command"
            ])),
            default_write_scope=list(data.get("default_write_scope", [])),
            command_allowlist=list(data.get("command_allowlist", [])),
            max_turns=int(data.get("max_turns", 12)),
            max_tool_calls=int(data.get("max_tool_calls", 20)),
            token_budget=int(data.get("token_budget", 25000)),
            minutes_budget=float(data.get("minutes_budget", 10.0)),
        )


class RoleRegistry:
    """Registry managing available specialist roles from configuration."""

    def __init__(self, config_path: str | Path = DEFAULT_ROLES_PATH) -> None:
        self.config_path = Path(config_path)
        self._roles: dict[str, RoleSpec] = {}
        self.load()

    def load(self) -> None:
        """Load roles from YAML file, falling back to safe defaults if not found."""
        if not self.config_path.exists():
            logger.warning("Roles config not found at %s. Using default backend and tester roles.", self.config_path)
            self._roles = {
                "backend": RoleSpec(
                    name="backend",
                    enabled=True,
                    description="Backend implementation",
                    default_write_scope=["*.py", "src/**"],
                    command_allowlist=["python", "python3", "pytest", "ruff", "mypy"],
                ),
                "tester": RoleSpec(
                    name="tester",
                    enabled=True,
                    description="Unit and integration tests",
                    default_write_scope=["tests/**", "test_*.py"],
                    command_allowlist=["pytest", "python", "python3"],
                ),
            }
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            roles_data = data.get("roles", {})
            self._roles = {}
            for name, role_dict in roles_data.items():
                self._roles[name] = RoleSpec.from_dict(name, role_dict or {})
        except Exception as exc:
            logger.error("Failed to parse roles config from %s: %s", self.config_path, exc)
            raise

    def get_role(self, name: str) -> RoleSpec:
        """Get role specification by name, raising KeyError if not defined."""
        if name not in self._roles:
            raise KeyError(f"Role '{name}' is not defined. Available: {list(self._roles.keys())}")
        return self._roles[name]

    def is_role_enabled(self, name: str) -> bool:
        """Check if role exists and is enabled."""
        role = self._roles.get(name)
        return role is not None and role.enabled

    def get_enabled_roles(self) -> dict[str, RoleSpec]:
        """Return mapping of currently enabled roles."""
        return {k: v for k, v in self._roles.items() if v.enabled}


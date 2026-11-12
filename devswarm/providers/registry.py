"""
devswarm.providers.registry — Provider factory and role resolution.

Resolves provider instances and role bindings from config/providers.yaml and
config/roles.yaml. Supports --offline fallback to local Ollama models.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from devswarm.providers.anthropic_adapter import AnthropicAdapter
from devswarm.providers.base import BaseProvider, ModelPricing
from devswarm.providers.mock_provider import MockProvider
from devswarm.providers.ollama_adapter import OllamaAdapter
from devswarm.providers.openai_adapter import OpenAICompatibleAdapter

logger = logging.getLogger(__name__)

DEFAULT_PROVIDERS_CONFIG = Path("config/providers.yaml")
DEFAULT_ROLES_CONFIG = Path("config/roles.yaml")


class ProviderRegistry:
    """Factory and registry for configured LLM providers."""

    def __init__(
        self,
        providers_config_path: Path | str = DEFAULT_PROVIDERS_CONFIG,
        roles_config_path: Path | str = DEFAULT_ROLES_CONFIG,
        offline: bool = False,
        mock_mode: bool = False,
    ) -> None:
        self.providers_config_path = Path(providers_config_path)
        self.roles_config_path = Path(roles_config_path)
        self.offline = offline
        self.mock_mode = mock_mode

        self.providers_config: dict[str, Any] = {}
        self.models_catalog: dict[str, Any] = {}
        self.roles_config: dict[str, Any] = {}
        self._instances: dict[str, BaseProvider] = {}

        self.load()

    def load(self) -> None:
        """Load providers and roles YAML configurations."""
        if self.providers_config_path.exists():
            try:
                data = yaml.safe_load(self.providers_config_path.read_text(encoding="utf-8")) or {}
                self.providers_config = data.get("providers", {})
                self.models_catalog = data.get("models", {})
            except Exception as exc:
                logger.warning("Failed loading providers config from %s: %s", self.providers_config_path, exc)

        if self.roles_config_path.exists():
            try:
                rdata = yaml.safe_load(self.roles_config_path.read_text(encoding="utf-8")) or {}
                self.roles_config = rdata.get("roles", {})
            except Exception as exc:
                logger.warning("Failed loading roles config from %s: %s", self.roles_config_path, exc)

    def register_provider(self, name: str, provider: BaseProvider) -> None:
        """Manually register a provider instance (useful for tests and mocks)."""
        self._instances[name] = provider

    def get_provider(self, name: str) -> BaseProvider:
        """Get or lazily instantiate provider by name."""
        if self.mock_mode:
            if name not in self._instances:
                self._instances[name] = MockProvider(name=name)
            return self._instances[name]

        if name in self._instances:
            return self._instances[name]

        cfg = self.providers_config.get(name, {})
        p_type = cfg.get("type", name)
        base_url = cfg.get("base_url", "")
        key_env = cfg.get("api_key_env", "")
        api_key = os.getenv(key_env) if key_env else None
        timeout = float(cfg.get("default_timeout_s", 60.0))
        env_timeout = os.getenv(f"{name.upper()}_TIMEOUT") or os.getenv("PROVIDER_TIMEOUT")
        if env_timeout:
            try:
                timeout = float(env_timeout)
            except ValueError:
                pass
        max_retries = int(cfg.get("max_retries", 3))

        # Instantiate provider adapter
        inst: BaseProvider
        if p_type == "ollama":
            inst = OllamaAdapter(base_url=base_url or "http://127.0.0.1:11434", default_timeout_s=timeout)
        elif p_type == "anthropic":
            inst = AnthropicAdapter(
                base_url=base_url or "https://api.anthropic.com/v1",
                api_key=api_key,
                default_timeout_s=timeout,
                max_retries=max_retries,
            )
        elif p_type in ("openai", "openai_compatible", "deepseek", "gemini"):
            inst = OpenAICompatibleAdapter(
                name=name,
                base_url=base_url or "https://api.openai.com/v1",
                api_key=api_key,
                default_timeout_s=timeout,
                max_retries=max_retries,
                models_catalog=self.models_catalog,
            )
        else:
            logger.warning("Unknown provider type '%s' for '%s'. Falling back to OpenAI compatible.", p_type, name)
            inst = OpenAICompatibleAdapter(
                name=name,
                base_url=base_url,
                api_key=api_key,
                default_timeout_s=timeout,
                models_catalog=self.models_catalog,
            )

        self._instances[name] = inst
        return inst

    def get_default_provider(self) -> BaseProvider:
        """Get the default provider (respects ORCHESTRATOR_PROVIDER env var, defaulting to siliconflow)."""
        provider_name = os.getenv("ORCHESTRATOR_PROVIDER", "siliconflow")
        return self.get_provider(provider_name)

    def get_role_binding(self, role: str) -> tuple[BaseProvider, str, bool, int, list[str]]:
        """
        Resolves the provider, model, thinking toggle, max_tokens, and allowed content classes for a role.
        Applies --offline fallback if active.
        """
        role_cfg = self.roles_config.get(role, {})
        target_provider = role_cfg.get("provider", "ollama")
        target_model = role_cfg.get("model", "")
        thinking = bool(role_cfg.get("thinking", False))
        max_tokens = int(role_cfg.get("max_tokens", 2048))
        content_classes = list(role_cfg.get("allowed_content_classes", ["*"]))

        # Check offline override
        if self.offline and target_provider.lower() != "ollama":
            logger.warning(
                "Role '%s' configured for cloud provider '%s' (%s), but --offline is active. "
                "Falling back to local Ollama (expect reduced reasoning/spec quality).",
                role, target_provider, target_model,
            )
            target_provider = "ollama"
            target_model = os.getenv("DEFAULT_LOCAL_MODEL", "qwen2.5-coder:7b")

        # Resolve placeholder <set model id>
        if not target_model or target_model == "<set model id>":
            if target_provider.lower() == "ollama":
                target_model = "qwen2.5-coder:7b"
            elif target_provider.lower() == "anthropic":
                target_model = "claude-3-7-sonnet"
            elif target_provider.lower() == "openai":
                target_model = "gpt-4o"
            elif target_provider.lower() == "gemini":
                target_model = "gemini-2.5-pro"
            elif target_provider.lower() == "deepseek":
                target_model = "deepseek-chat"

        provider_inst = self.get_provider(target_provider)
        return provider_inst, target_model, thinking, max_tokens, content_classes


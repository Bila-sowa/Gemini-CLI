"""Supported Gemini models, tool declarations, and fallback policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ModelTier(StrEnum):
    """Coarse capability class used for documentation and selection UIs."""

    FLASH = "flash"
    FLASH_LITE = "flash_lite"
    PRO = "pro"


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a supported Gemini model."""

    name: str
    display_name: str
    tier: ModelTier
    context_window: int
    max_output_tokens: int
    description: str
    fallback: str | None = None


# Registry of all known text models, newest first. ``fallback`` names another
# registry entry that is tried when the primary model fails with an
# unrecoverable error; the lookup follows exactly one link.
#
# Limits are taken from the Gemini API model pages (November 2026). The 2.5
# entries are kept for existing configurations: Google has deprecated them and
# routes new API traffic to the models listed first.
MODEL_REGISTRY: dict[str, ModelInfo] = {
    "gemini-4-argon": ModelInfo(
        name="gemini-4-argon",
        display_name="Gemini 4 Argon",
        tier=ModelTier.PRO,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description=(
            "Flagship reasoning for complex software engineering, science, and "
            "long agent runs."
        ),
        fallback="gemini-3.8-flash",
    ),
    "gemini-3.8-flash": ModelInfo(
        name="gemini-3.8-flash",
        display_name="Gemini 3.8 Flash",
        tier=ModelTier.FLASH,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description=(
            "Most intelligent Flash model for long-horizon coding and "
            "autonomous agents."
        ),
        fallback="gemini-3.6-flash",
    ),
    "gemini-3.6-flash": ModelInfo(
        name="gemini-3.6-flash",
        display_name="Gemini 3.6 Flash",
        tier=ModelTier.FLASH,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description=(
            "Previous-generation Flash for agentic coding and everyday tasks."
        ),
        fallback="gemini-3.5-flash-lite",
    ),
    "gemini-3.5-flash-lite": ModelInfo(
        name="gemini-3.5-flash-lite",
        display_name="Gemini 3.5 Flash-Lite",
        tier=ModelTier.FLASH_LITE,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description=(
            "Fast, low-cost 3.5 model for high-throughput and subagent tasks."
        ),
        fallback="gemini-3.1-flash-lite",
    ),
    "gemini-3.1-flash-lite": ModelInfo(
        name="gemini-3.1-flash-lite",
        display_name="Gemini 3.1 Flash-Lite",
        tier=ModelTier.FLASH_LITE,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description=("Low-latency 3.1 model for high-frequency lightweight tasks."),
    ),
    "gemini-3.1-pro-preview": ModelInfo(
        name="gemini-3.1-pro-preview",
        display_name="Gemini 3.1 Pro Preview",
        tier=ModelTier.PRO,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description=(
            "Preview Pro model for precise tool use and multi-step agentic workflows."
        ),
        fallback="gemini-3.8-flash",
    ),
    "gemini-3-flash-preview": ModelInfo(
        name="gemini-3-flash-preview",
        display_name="Gemini 3 Flash Preview",
        tier=ModelTier.FLASH,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description=(
            "Preview Flash with strong multimodal understanding and agentic "
            "coding skills."
        ),
        fallback="gemini-3.8-flash",
    ),
    "gemini-2.5-pro": ModelInfo(
        name="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro (legacy)",
        tier=ModelTier.PRO,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description=(
            "Legacy Pro model with limited API access; prefer Gemini 4 Argon."
        ),
        fallback="gemini-3.8-flash",
    ),
    "gemini-2.5-flash": ModelInfo(
        name="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash (legacy)",
        tier=ModelTier.FLASH,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description=(
            "Legacy Flash model with limited API access; prefer Gemini 3.8 Flash."
        ),
        fallback="gemini-3.5-flash-lite",
    ),
}


class UnknownModelError(ValueError):
    """Raised when a model name is not present in :data:`MODEL_REGISTRY`."""

    def __init__(self, name: str) -> None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        super().__init__(f"Unknown model '{name}'. Available: {available}")
        self.name = name


@dataclass(frozen=True)
class ToolDefinition:
    """Model-facing declaration of a tool the agent may call.

    The class deliberately contains no SDK types: the LLM client translates it
    into the provider representation. ``parameters_json_schema`` follows JSON
    Schema, as produced by ``ToolRegistry.export_schemas()``.
    """

    name: str
    description: str
    parameters_json_schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_export(cls, payload: Mapping[str, Any]) -> ToolDefinition:
        """Build a definition from a registry export entry.

        Raises:
            ValueError: if ``name`` is missing or empty.
        """
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            msg = "Tool definition is missing a non-empty 'name'."
            raise ValueError(msg)

        description = payload.get("description")
        schema = payload.get("parameters_json_schema", payload.get("parameters"))
        return cls(
            name=name,
            description=description if isinstance(description, str) else "",
            parameters_json_schema=dict(schema) if isinstance(schema, Mapping) else {},
        )


def get_model(name: str) -> ModelInfo:
    """Look up a model by its API name.

    Raises:
        UnknownModelError: if the model is not registered.
    """
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise UnknownModelError(name) from None


def get_fallback(name: str) -> ModelInfo | None:
    """Return the model to try when *name* fails unrecoverably.

    The lookup follows exactly one fallback link, so a misconfigured registry
    cannot trap the caller in an endless chain.

    Raises:
        UnknownModelError: if *name* is not registered.
    """
    model = get_model(name)
    if model.fallback is None:
        return None
    return get_model(model.fallback)


def list_models() -> list[ModelInfo]:
    """Return all supported models in registry order (newest first)."""
    return list(MODEL_REGISTRY.values())


__all__ = [
    "MODEL_REGISTRY",
    "ModelInfo",
    "ModelTier",
    "ToolDefinition",
    "UnknownModelError",
    "get_fallback",
    "get_model",
    "list_models",
]

"""Supported Gemini and Gemma models, tool declarations, and fallback policy."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

# Conservative metadata for a model that is configured by hand but absent from
# the registry. The limits are the largest published Gemini 4 window, so a
# custom model is never rejected locally for being too small.
CUSTOM_MODEL_CONTEXT_WINDOW = 1_048_576
CUSTOM_MODEL_MAX_OUTPUT_TOKENS = 65_536

# Custom models are resolved for every request; warn once per name so an
# unsupported hand-written value is visible without flooding the log.
_warned_custom_models: set[str] = set()


class ModelTier(StrEnum):
    """Coarse capability class used for documentation and selection UIs."""

    FLASH = "flash"
    FLASH_LITE = "flash_lite"
    PRO = "pro"
    GEMMA = "gemma"


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a supported model."""

    name: str
    display_name: str
    tier: ModelTier
    context_window: int
    max_output_tokens: int
    description: str
    fallback: str | None = None


# Registry of known text models, newest first. ``fallback`` points to another
# registry entry and is followed at most once. Custom models are resolved by
# :func:`resolve_model` and deliberately have no fallback.
#
# Limits are taken from the published model cards (November 2026). Gemma
# entries use the open-model context and output limits. The 2.5 entries remain
# available for existing users but are marked as legacy.
MODEL_REGISTRY: dict[str, ModelInfo] = {
    "gemini-4-argon": ModelInfo(
        name="gemini-4-argon",
        display_name="Gemini 4 Argon",
        tier=ModelTier.PRO,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description=(
            "Flagship reasoning for complex engineering, science, and long agent runs."
        ),
        fallback="gemini-3.8-flash",
    ),
    "gemini-4-carbon": ModelInfo(
        name="gemini-4-carbon",
        display_name="Gemini 4 Carbon",
        tier=ModelTier.PRO,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description=(
            "Balanced Gemini 4 Pro model for agentic coding and long-context workflows."
        ),
        fallback="gemini-4-hydrogen",
    ),
    "gemini-4-neon": ModelInfo(
        name="gemini-4-neon",
        display_name="Gemini 4 Neon",
        tier=ModelTier.PRO,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description=(
            "Fast Gemini 4 Pro model for high-throughput agents and precise tool use."
        ),
        fallback="gemini-4-hydrogen",
    ),
    "gemini-4-hydrogen": ModelInfo(
        name="gemini-4-hydrogen",
        display_name="Gemini 4 Hydrogen",
        tier=ModelTier.FLASH,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description=(
            "Efficient Gemini 4 Flash model for everyday agent tasks and subagents."
        ),
        fallback="gemini-3.8-flash",
    ),
    "gemma-4-31b-it": ModelInfo(
        name="gemma-4-31b-it",
        display_name="Gemma 4 31B",
        tier=ModelTier.GEMMA,
        context_window=131_072,
        max_output_tokens=8_192,
        description=("Open Gemma 4 model for local-style agent work and tool calling."),
    ),
    "gemma-4-26b-a4b-it": ModelInfo(
        name="gemma-4-26b-a4b-it",
        display_name="Gemma 4 26B",
        tier=ModelTier.GEMMA,
        context_window=131_072,
        max_output_tokens=8_192,
        description=(
            "Efficient open Gemma 4 mixture-of-experts model for agent tasks."
        ),
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
    "gemini-3.7-flash": ModelInfo(
        name="gemini-3.7-flash",
        display_name="Gemini 3.7 Flash",
        tier=ModelTier.FLASH,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description=("Previous Flash generation for reliable everyday agent tasks."),
        fallback="gemini-3.8-flash",
    ),
    "gemini-3.6-flash": ModelInfo(
        name="gemini-3.6-flash",
        display_name="Gemini 3.6 Flash",
        tier=ModelTier.FLASH,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description="Previous-generation Flash for agentic coding and everyday tasks.",
        fallback="gemini-3.5-flash-lite",
    ),
    "gemini-3.5-flash": ModelInfo(
        name="gemini-3.5-flash",
        display_name="Gemini 3.5 Flash",
        tier=ModelTier.FLASH,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description="Older Flash generation for general assistant and tool tasks.",
        fallback="gemini-3.6-flash",
    ),
    "gemini-3.5-flash-lite": ModelInfo(
        name="gemini-3.5-flash-lite",
        display_name="Gemini 3.5 Flash-Lite",
        tier=ModelTier.FLASH_LITE,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description="Fast, low-cost 3.5 model for high-throughput and subagent tasks.",
        fallback="gemini-3.1-flash-lite",
    ),
    "gemini-3.1-flash-lite": ModelInfo(
        name="gemini-3.1-flash-lite",
        display_name="Gemini 3.1 Flash-Lite",
        tier=ModelTier.FLASH_LITE,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description="Low-latency 3.1 model for high-frequency lightweight tasks.",
    ),
    "gemini-3.1-pro-preview": ModelInfo(
        name="gemini-3.1-pro-preview",
        display_name="Gemini 3.1 Pro Preview",
        tier=ModelTier.PRO,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description="Preview Pro for precise tool use and multi-step workflows.",
        fallback="gemini-3.8-flash",
    ),
    "gemini-3-flash-preview": ModelInfo(
        name="gemini-3-flash-preview",
        display_name="Gemini 3 Flash Preview",
        tier=ModelTier.FLASH,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description="Preview Flash with strong multimodal and agentic coding skills.",
        fallback="gemini-3.8-flash",
    ),
    "gemini-2.5-pro": ModelInfo(
        name="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro (legacy)",
        tier=ModelTier.PRO,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description="Legacy Pro model; prefer Gemini 4 Argon.",
        fallback="gemini-3.8-flash",
    ),
    "gemini-2.5-flash": ModelInfo(
        name="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash (legacy)",
        tier=ModelTier.FLASH,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description="Legacy Flash model; prefer Gemini 3.8 Flash.",
        fallback="gemini-3.5-flash-lite",
    ),
    "gemini-2.5-flash-lite": ModelInfo(
        name="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash-Lite (legacy)",
        tier=ModelTier.FLASH_LITE,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description="Legacy low-cost model; prefer Gemini 3.5 Flash-Lite.",
        fallback="gemini-3.5-flash-lite",
    ),
}


class UnknownModelError(ValueError):
    """Raised when a model name is empty, malformed, or not registered."""

    def __init__(self, name: str) -> None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        super().__init__(f"Unknown model '{name}'. Available: {available}")
        self.name = name


@dataclass(frozen=True)
class ToolDefinition:
    """A tool declaration the model may call.

    The class deliberately avoids SDK types. ``parameters_json_schema`` follows
    JSON Schema, as produced by ``ToolRegistry.export_schemas()``.
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


def _is_valid_custom_model_name(name: str) -> bool:
    """Return whether *name* is safe to send to the provider as-is."""
    if not name or len(name) > 256:
        return False
    return all(
        character.isprintable() and not character.isspace() for character in name
    )


def get_model(name: str) -> ModelInfo:
    """Look up a registered model by its API name.

    Raises:
        UnknownModelError: if the model is not registered.
    """
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise UnknownModelError(name) from None


def resolve_model(name: str) -> ModelInfo:
    """Resolve a configured model name, accepting well-formed custom models.

    Registered models keep their published metadata. A non-empty, printable
    name without whitespace that is not in the registry is treated as a custom
    model with conservative generic metadata and no fallback. This keeps a
    hand-written ``.env`` value from stopping the application at startup; the
    provider still reports an error if the model does not exist.

    Raises:
        UnknownModelError: if *name* is empty or contains whitespace or control
            characters.
    """
    if not _is_valid_custom_model_name(name):
        raise UnknownModelError(name)

    known = MODEL_REGISTRY.get(name)
    if known is not None:
        return known

    if name not in _warned_custom_models:
        _warned_custom_models.add(name)
        logger.warning("Using custom model %r with generic metadata.", name)
    return ModelInfo(
        name=name,
        display_name=name,
        tier=ModelTier.FLASH,
        context_window=CUSTOM_MODEL_CONTEXT_WINDOW,
        max_output_tokens=CUSTOM_MODEL_MAX_OUTPUT_TOKENS,
        description="Custom model from configuration; metadata is generic.",
        fallback=None,
    )


def get_fallback(name: str) -> ModelInfo | None:
    """Resolve the single fallback link for *name*.

    Raises:
        UnknownModelError: if *name* or its fallback is not registered.
    """
    model = get_model(name)
    if model.fallback is None:
        return None
    return get_model(model.fallback)


def list_models() -> list[ModelInfo]:
    """Return all registered models in registry order."""
    return list(MODEL_REGISTRY.values())


__all__ = [
    "CUSTOM_MODEL_CONTEXT_WINDOW",
    "CUSTOM_MODEL_MAX_OUTPUT_TOKENS",
    "MODEL_REGISTRY",
    "ModelInfo",
    "ModelTier",
    "ToolDefinition",
    "UnknownModelError",
    "get_fallback",
    "get_model",
    "list_models",
    "resolve_model",
]

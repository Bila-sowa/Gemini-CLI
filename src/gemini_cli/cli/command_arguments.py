"""Validation and suggestions for slash-command arguments."""

from __future__ import annotations

import difflib
from collections.abc import Sequence

from gemini_cli.config.defaults import AVAILABLE_ROLES, AVAILABLE_THEMES
from gemini_cli.llm import MODEL_REGISTRY


class CommandArgumentError(ValueError):
    """Raised when a slash-command argument is invalid."""


def _suggest(value: str, choices: Sequence[str]) -> str | None:
    """Return the closest choice without executing anything."""
    matches = difflib.get_close_matches(value, list(choices), n=1, cutoff=0.6)
    return matches[0] if matches else None


def parse_model_argument(raw: str) -> str:
    """Validate a model name; only registered models are accepted.

    Custom models remain available through ``GEMINI_MODEL`` or ``--model``.
    """
    candidate = raw.strip().lower()
    if not candidate:
        msg = "Model name cannot be empty."
        raise CommandArgumentError(msg)

    if candidate in MODEL_REGISTRY:
        return candidate

    suggestion = _suggest(candidate, tuple(MODEL_REGISTRY))
    if suggestion is not None:
        msg = f"Unknown model '{raw.strip()}'. Did you mean '{suggestion}'?"
        raise CommandArgumentError(msg)

    msg = (
        f"Unknown model '{raw.strip()}'. "
        "Custom models must be set with GEMINI_MODEL or --model."
    )
    raise CommandArgumentError(msg)


def parse_role_argument(raw: str) -> str:
    """Validate a role name against the configured roles."""
    candidate = raw.strip().lower()
    if not candidate:
        msg = "Role name cannot be empty."
        raise CommandArgumentError(msg)

    if candidate in AVAILABLE_ROLES:
        return candidate

    suggestion = _suggest(candidate, AVAILABLE_ROLES)
    available = ", ".join(AVAILABLE_ROLES)
    if suggestion is not None:
        msg = (
            f"Unknown role '{raw.strip()}'. "
            f"Did you mean '{suggestion}'? Available: {available}."
        )
        raise CommandArgumentError(msg)

    msg = f"Unknown role '{raw.strip()}'. Available: {available}."
    raise CommandArgumentError(msg)


def parse_theme_argument(raw: str) -> str:
    """Validate a theme name against the available palettes."""
    candidate = raw.strip().lower()
    if not candidate:
        msg = "Theme name cannot be empty."
        raise CommandArgumentError(msg)

    if candidate in AVAILABLE_THEMES:
        return candidate

    suggestion = _suggest(candidate, AVAILABLE_THEMES)
    available = ", ".join(AVAILABLE_THEMES)
    if suggestion is not None:
        msg = (
            f"Unknown theme '{raw.strip()}'. "
            f"Did you mean '{suggestion}'? Available: {available}."
        )
        raise CommandArgumentError(msg)

    msg = f"Unknown theme '{raw.strip()}'. Available: {available}."
    raise CommandArgumentError(msg)


__all__ = [
    "CommandArgumentError",
    "parse_model_argument",
    "parse_role_argument",
    "parse_theme_argument",
]

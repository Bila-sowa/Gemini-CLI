"""Tests for built-in configuration defaults."""

from __future__ import annotations

import pytest

from gemini_cli.config.defaults import (
    AVAILABLE_ROLES,
    DEFAULT_ALLOWED_EXTENSIONS,
    DEFAULT_MAX_AGENT_ITERATIONS,
    DEFAULT_MAX_FILE_SIZE_BYTES,
    DEFAULT_MAX_HISTORY_TURNS,
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REQUEST_TIMEOUT_MS,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_ROLE,
    DEFAULT_SHELL_MAX_OUTPUT_BYTES,
    DEFAULT_SHELL_TIMEOUT_SECONDS,
    DEFAULT_TELEMETRY_ENABLED,
    GLOBAL_CONFIG_DIR_NAME,
    GLOBAL_CONFIG_FILE_NAME,
    PROJECT_CONFIG_FILE_NAME,
    PROJECT_INSTRUCTION_FILES,
)


def test_default_model_is_non_empty() -> None:
    """The default model must be a non-empty string."""
    assert isinstance(DEFAULT_MODEL, str)
    assert DEFAULT_MODEL


@pytest.mark.parametrize(
    "value",
    [
        DEFAULT_REQUEST_TIMEOUT_MS,
        DEFAULT_MAX_HISTORY_TURNS,
        DEFAULT_MAX_INPUT_CHARS,
        DEFAULT_MAX_AGENT_ITERATIONS,
        DEFAULT_MAX_OUTPUT_TOKENS,
        DEFAULT_MAX_FILE_SIZE_BYTES,
        DEFAULT_SHELL_TIMEOUT_SECONDS,
        DEFAULT_SHELL_MAX_OUTPUT_BYTES,
    ],
)
def test_positive_numeric_defaults(value: int) -> None:
    """Limits and timeouts must be strictly positive."""
    assert value > 0


def test_retry_attempts_is_bounded() -> None:
    """Retry attempts must match the range accepted by the settings model."""
    assert 0 <= DEFAULT_RETRY_ATTEMPTS <= 10


def test_allowed_extensions_are_normalized() -> None:
    """Extensions must be unique, lowercase, and dot-prefixed."""
    assert isinstance(DEFAULT_ALLOWED_EXTENSIONS, tuple)
    assert DEFAULT_ALLOWED_EXTENSIONS
    assert len(set(DEFAULT_ALLOWED_EXTENSIONS)) == len(DEFAULT_ALLOWED_EXTENSIONS)

    for extension in DEFAULT_ALLOWED_EXTENSIONS:
        assert extension == extension.lower()
        assert extension == extension.strip()
        assert extension.startswith(".")

    assert {".py", ".md", ".json"} <= set(DEFAULT_ALLOWED_EXTENSIONS)


def test_roles_are_valid() -> None:
    """Roles must be unique, lowercase, and include the default role."""
    assert isinstance(AVAILABLE_ROLES, tuple)
    assert AVAILABLE_ROLES
    assert len(set(AVAILABLE_ROLES)) == len(AVAILABLE_ROLES)
    assert all(role == role.lower() for role in AVAILABLE_ROLES)
    assert DEFAULT_ROLE in AVAILABLE_ROLES


def test_instruction_files_are_declared() -> None:
    """Project instruction discovery must include the documented files."""
    assert isinstance(PROJECT_INSTRUCTION_FILES, tuple)
    assert "GEMINI.md" in PROJECT_INSTRUCTION_FILES
    assert "AGENTS.md" in PROJECT_INSTRUCTION_FILES
    assert "AGENT.md" in PROJECT_INSTRUCTION_FILES
    assert all(name.endswith(".md") for name in PROJECT_INSTRUCTION_FILES)


def test_telemetry_is_disabled_by_default() -> None:
    """Telemetry must be opt-in."""
    assert DEFAULT_TELEMETRY_ENABLED is False


def test_config_file_names() -> None:
    """Configuration locations must use the documented names."""
    assert GLOBAL_CONFIG_DIR_NAME == ".gemini-cli"
    assert GLOBAL_CONFIG_FILE_NAME == "config.toml"
    assert PROJECT_CONFIG_FILE_NAME == ".gemini-cli.toml"


def test_collections_are_immutable() -> None:
    """Module-level collections must not be mutable."""
    assert isinstance(DEFAULT_ALLOWED_EXTENSIONS, tuple)
    assert isinstance(PROJECT_INSTRUCTION_FILES, tuple)
    assert isinstance(AVAILABLE_ROLES, tuple)

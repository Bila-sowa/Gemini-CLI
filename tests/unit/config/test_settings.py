"""Tests for typed application settings and their layered precedence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from gemini_cli.config import Settings
from gemini_cli.config.defaults import (
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
)


@pytest.fixture(autouse=True)
def _isolated_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep tests away from real env vars, config files, and home directory."""
    for name in list(os.environ):
        if name.startswith("GEMINI_"):
            monkeypatch.delenv(name, raising=False)

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(tmp_path)


def make_settings(**overrides: Any) -> Settings:
    """Build settings without reading a developer's local ``.env`` file."""
    return Settings(_env_file=None, **overrides)


def write_toml(path: Path, content: str) -> Path:
    """Write a TOML file, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_defaults_apply_without_configuration() -> None:
    settings = make_settings()

    assert settings.api_key == ""
    assert settings.model == DEFAULT_MODEL
    assert settings.request_timeout_ms == DEFAULT_REQUEST_TIMEOUT_MS
    assert settings.retry_attempts == DEFAULT_RETRY_ATTEMPTS
    assert settings.max_history_turns == DEFAULT_MAX_HISTORY_TURNS
    assert settings.max_input_chars == DEFAULT_MAX_INPUT_CHARS
    assert settings.max_agent_iterations == DEFAULT_MAX_AGENT_ITERATIONS
    assert settings.max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS
    assert settings.allowed_roots == []
    assert settings.max_file_size_bytes == DEFAULT_MAX_FILE_SIZE_BYTES
    assert settings.shell_timeout_seconds == DEFAULT_SHELL_TIMEOUT_SECONDS
    assert settings.shell_max_output_bytes == DEFAULT_SHELL_MAX_OUTPUT_BYTES
    assert settings.role == DEFAULT_ROLE
    assert settings.telemetry_enabled is DEFAULT_TELEMETRY_ENABLED


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    monkeypatch.setenv("GEMINI_REQUEST_TIMEOUT_MS", "1234")
    monkeypatch.setenv("GEMINI_RETRY_ATTEMPTS", "0")
    monkeypatch.setenv("GEMINI_ROLE", "  CODE  ")
    monkeypatch.setenv("GEMINI_TELEMETRY_ENABLED", "true")

    settings = make_settings()

    assert settings.model == "gemini-2.5-pro"
    assert settings.request_timeout_ms == 1234
    assert settings.retry_attempts == 0
    assert settings.role == "code"
    assert settings.telemetry_enabled is True


def test_constructor_arguments_override_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "from-env")

    settings = make_settings(model="from-argument")

    assert settings.model == "from-argument"


def test_dotenv_file_is_loaded(tmp_path: Path) -> None:
    dotenv = tmp_path / "custom.env"
    dotenv.write_text("GEMINI_MODEL=from-dotenv\n", encoding="utf-8")

    settings = Settings(_env_file=dotenv)

    assert settings.model == "from-dotenv"


def test_dotenv_overrides_project_toml(tmp_path: Path) -> None:
    write_toml(
        tmp_path / ".gemini-cli.toml",
        'model = "from-project"\n',
    )
    (tmp_path / ".env").write_text("GEMINI_MODEL=from-dotenv\n", encoding="utf-8")

    settings = Settings()

    assert settings.model == "from-dotenv"


def test_environment_overrides_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text("GEMINI_MODEL=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_MODEL", "from-env")

    settings = Settings()

    assert settings.model == "from-env"


def test_project_toml_overrides_defaults(tmp_path: Path) -> None:
    write_toml(
        tmp_path / ".gemini-cli.toml",
        "\n".join(
            [
                'model = "gemini-2.5-pro"',
                "retry_attempts = 7",
                "max_history_turns = 5",
                'role = "analyser"',
                "telemetry_enabled = true",
            ]
        ),
    )

    settings = make_settings()

    assert settings.model == "gemini-2.5-pro"
    assert settings.retry_attempts == 7
    assert settings.max_history_turns == 5
    assert settings.role == "analyser"
    assert settings.telemetry_enabled is True


def test_global_toml_is_used_when_project_file_is_missing(tmp_path: Path) -> None:
    write_toml(
        Path.home() / ".gemini-cli" / "config.toml",
        'model = "gemini-from-global"\n',
    )

    settings = make_settings()

    assert settings.model == "gemini-from-global"
    assert not (tmp_path / ".gemini-cli.toml").exists()


def test_project_toml_overrides_global_toml(tmp_path: Path) -> None:
    write_toml(
        Path.home() / ".gemini-cli" / "config.toml",
        'model = "from-global"\nretry_attempts = 1\n',
    )
    write_toml(
        tmp_path / ".gemini-cli.toml",
        'model = "from-project"\n',
    )

    settings = make_settings()

    assert settings.model == "from-project"
    assert settings.retry_attempts == 1


def test_environment_overrides_project_toml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_toml(tmp_path / ".gemini-cli.toml", 'model = "from-project"\n')
    monkeypatch.setenv("GEMINI_MODEL", "from-env")

    settings = make_settings()

    assert settings.model == "from-env"


def test_project_toml_cannot_set_blocked_fields(tmp_path: Path) -> None:
    write_toml(
        tmp_path / ".gemini-cli.toml",
        "\n".join(
            [
                'api_key = "project-injected-key"',
                'allowed_roots = ["/"]',
            ]
        ),
    )

    settings = make_settings()

    assert settings.api_key == ""
    assert settings.allowed_roots == []


def test_global_toml_may_set_allowed_roots() -> None:
    write_toml(
        Path.home() / ".gemini-cli" / "config.toml",
        'allowed_roots = ["/tmp/approved"]\n',
    )

    settings = make_settings()

    assert settings.allowed_roots == [Path("/tmp/approved")]


def test_allowed_roots_are_parsed_from_json_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_ALLOWED_ROOTS", '["/tmp/a", "/tmp/b"]')

    settings = make_settings()

    assert settings.allowed_roots == [Path("/tmp/a"), Path("/tmp/b")]


def test_unknown_environment_variables_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_UNKNOWN_SETTING", "ignored")

    settings = make_settings()

    assert settings.model == DEFAULT_MODEL


def test_invalid_role_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="Unknown role"):
        make_settings(role="nonexistent")


def test_api_key_may_be_empty() -> None:
    settings = make_settings()

    assert settings.api_key == ""


def test_api_key_is_not_exposed_in_repr() -> None:
    settings = make_settings(api_key="super-secret-value")

    assert "super-secret-value" not in repr(settings)


@pytest.mark.parametrize("value", [0, -1])
def test_request_timeout_must_be_positive(value: int) -> None:
    with pytest.raises(ValidationError):
        make_settings(request_timeout_ms=value)


@pytest.mark.parametrize("value", [-1, 11])
def test_retry_attempts_are_bounded(value: int) -> None:
    with pytest.raises(ValidationError):
        make_settings(retry_attempts=value)


@pytest.mark.parametrize("value", [0, 10])
def test_retry_attempts_accepts_boundary_values(value: int) -> None:
    assert make_settings(retry_attempts=value).retry_attempts == value


def test_invalid_project_toml_value_raises_validation_error(
    tmp_path: Path,
) -> None:
    write_toml(tmp_path / ".gemini-cli.toml", "request_timeout_ms = 0\n")

    with pytest.raises(ValidationError):
        make_settings()

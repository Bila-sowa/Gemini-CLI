"""Typed application settings with layered precedence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from gemini_cli.config.defaults import (
    AVAILABLE_ROLES,
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
)

logger = logging.getLogger(__name__)

# Project configuration files are untrusted input: they live in a checkout that
# may come from anywhere. They may tune behavior, but must not widen filesystem
# access or inject credentials.
PROJECT_CONFIG_BLOCKED_FIELDS: frozenset[str] = frozenset({"api_key", "allowed_roots"})


def project_config_path() -> Path:
    """Return the project-level configuration file for the current directory."""
    return Path.cwd() / PROJECT_CONFIG_FILE_NAME


def global_config_path() -> Path | None:
    """Return the user-level configuration file, if a home directory exists."""
    try:
        home = Path.home()
    except RuntimeError:
        logger.warning("Cannot determine home directory; global config is ignored.")
        return None
    return home / GLOBAL_CONFIG_DIR_NAME / GLOBAL_CONFIG_FILE_NAME


class _ProjectTomlConfigSettingsSource(TomlConfigSettingsSource):
    """Project TOML source that drops security-sensitive keys."""

    def __call__(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.init_kwargs.items()
            if key not in PROJECT_CONFIG_BLOCKED_FIELDS
        }


class Settings(BaseSettings):
    """Application-wide settings validated at startup.

    Values are resolved with the following precedence (highest first):

    1. Explicit constructor arguments.
    2. Environment variables (prefixed with ``GEMINI_``).
    3. A ``.env`` file in the working directory.
    4. Secrets directory (when configured).
    5. Project-level ``.gemini-cli.toml`` in the working directory.
    6. Global ``~/.gemini-cli/config.toml``.
    7. Built-in defaults from :mod:`gemini_cli.config.defaults`.

    The project-level file may not set ``api_key`` or ``allowed_roots``;
    see :data:`PROJECT_CONFIG_BLOCKED_FIELDS`. TOML files use field names,
    for example ``model = "gemini-2.5-pro"``.
    """

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # --- API ---
    api_key: str = Field(default="", repr=False, description="Gemini API key.")
    model: str = Field(default=DEFAULT_MODEL)
    request_timeout_ms: int = Field(default=DEFAULT_REQUEST_TIMEOUT_MS, gt=0)
    retry_attempts: int = Field(default=DEFAULT_RETRY_ATTEMPTS, ge=0, le=10)

    # --- Conversation ---
    max_history_turns: int = Field(default=DEFAULT_MAX_HISTORY_TURNS, ge=1)
    max_input_chars: int = Field(default=DEFAULT_MAX_INPUT_CHARS, ge=100)
    max_agent_iterations: int = Field(default=DEFAULT_MAX_AGENT_ITERATIONS, ge=1)
    max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, ge=1)

    # --- Filesystem ---
    allowed_roots: list[Path] = Field(default_factory=list)
    max_file_size_bytes: int = Field(default=DEFAULT_MAX_FILE_SIZE_BYTES, gt=0)

    # --- Shell ---
    shell_timeout_seconds: int = Field(default=DEFAULT_SHELL_TIMEOUT_SECONDS, gt=0)
    shell_max_output_bytes: int = Field(default=DEFAULT_SHELL_MAX_OUTPUT_BYTES, gt=0)

    # --- Role ---
    role: str = Field(default=DEFAULT_ROLE)

    # --- Telemetry ---
    telemetry_enabled: bool = Field(default=DEFAULT_TELEMETRY_ENABLED)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        """Normalize the role name and reject unknown values."""
        normalized = value.strip().lower()
        if normalized not in AVAILABLE_ROLES:
            msg = (
                f"Unknown role '{normalized}'. Available: {', '.join(AVAILABLE_ROLES)}"
            )
            raise ValueError(msg)
        return normalized

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Register custom sources; earlier entries take precedence."""
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            _ProjectTomlConfigSettingsSource(
                settings_cls,
                toml_file=project_config_path(),
            ),
        ]

        global_path = global_config_path()
        if global_path is not None:
            sources.append(
                TomlConfigSettingsSource(settings_cls, toml_file=global_path)
            )

        return tuple(sources)

"""Configuration commands."""

from __future__ import annotations

import json
import os
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError

from gemini_cli.config import Settings
from gemini_cli.config.defaults import PROJECT_CONFIG_FILE_NAME
from gemini_cli.config.settings import global_config_path

_ERROR_COLOR = typer.colors.RED
_BLOCKED_FIELDS = frozenset({"api_key", "allowed_roots"})
_BOOL_FIELDS = frozenset({"telemetry_enabled"})
_INT_FIELDS = frozenset(
    {
        "max_agent_iterations",
        "max_file_size_bytes",
        "max_history_turns",
        "max_input_chars",
        "max_output_tokens",
        "quota_5h_token_limit",
        "quota_weekly_token_limit",
        "request_timeout_ms",
        "retry_attempts",
        "shell_max_output_bytes",
        "shell_timeout_seconds",
    }
)


def _print_error(message: str) -> None:
    typer.secho(f"Error: {message}", fg=_ERROR_COLOR, err=True)


def _display(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "(none)"
    return str(value)


def show() -> None:
    """Print all resolved settings with the API key redacted."""
    settings = Settings()

    typer.echo(f"api_key: {'set' if settings.api_key else 'not set'}")
    for name, value in settings.model_dump().items():
        if name == "api_key":
            continue
        typer.echo(f"{name}: {_display(value)}")


def set_value(
    key: str = typer.Argument(..., help="Setting name."),
    value: str = typer.Argument(..., help="New value."),
    global_: bool = typer.Option(
        False,
        "--global",
        help="Write to the user-level config instead of the project config.",
    ),
) -> None:
    """Write one setting to the project or global configuration."""
    if key in _BLOCKED_FIELDS:
        _print_error(
            f"'{key}' cannot be written by this command. "
            "Use the environment, the keyring, or the config file directly."
        )
        raise typer.Exit(code=1)

    if key not in Settings.model_fields:
        available = ", ".join(sorted(set(Settings.model_fields) - _BLOCKED_FIELDS))
        _print_error(f"unknown setting '{key}'. Available: {available}")
        raise typer.Exit(code=1)

    try:
        converted = _convert(key, value)
        candidate = Settings(**{**Settings().model_dump(), key: converted})
    except (ValueError, ValidationError) as exc:
        _print_error(f"invalid value for '{key}': {exc}")
        raise typer.Exit(code=1) from exc

    try:
        path = _target_path(global_)
        document = _read_document(path)
        document[key] = candidate.model_dump()[key]
        _write_document(path, document)
    except (OSError, ValueError) as exc:
        _print_error(str(exc))
        raise typer.Exit(code=1) from exc

    typer.echo(f"Set {key} = {_display(candidate.model_dump()[key])} in {path}")


def _convert(key: str, raw: str) -> object:
    text = raw.strip()
    if key in _BOOL_FIELDS:
        lowered = text.lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        msg = f"{key} must be true or false."
        raise ValueError(msg)
    if key in _INT_FIELDS:
        try:
            return int(text)
        except ValueError:
            msg = f"{key} must be an integer."
            raise ValueError(msg) from None
    if not text:
        msg = f"{key} must not be empty."
        raise ValueError(msg)
    return text


def _target_path(use_global: bool) -> Path:
    if not use_global:
        return Path.cwd() / PROJECT_CONFIG_FILE_NAME
    path = global_config_path()
    if path is None:
        msg = "Cannot determine the home directory for the global config."
        raise OSError(msg)
    return path


def _read_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        msg = f"Could not parse {path}: {exc}"
        raise ValueError(msg) from exc
    return document


def _write_document(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{name} = {_toml_value(value)}" for name, value in sorted(document.items())
    ]
    payload = "\n".join(lines) + "\n"

    descriptor, raw_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=".config-",
        suffix=".tmp",
    )
    temporary = Path(raw_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        items = ", ".join(_toml_value(item) for item in value)
        return f"[{items}]"
    msg = f"Cannot write value of type {type(value).__name__} to TOML."
    raise ValueError(msg)


__all__ = ["set_value", "show"]

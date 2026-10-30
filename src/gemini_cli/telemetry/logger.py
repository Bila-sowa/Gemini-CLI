"""Structured logging, secret redaction, and telemetry setup."""

from __future__ import annotations

import datetime
import json
import logging
import logging.handlers
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.logging import RichHandler

LOGGER_NAME = "gemini_cli"
TELEMETRY_LOGGER_NAME = "gemini_cli.telemetry"

_HANDLERS_MARKER = "_gemini_cli_managed"

_SENSITIVE_KEYS = r"(?:api[_-]?key|key|token|password|secret|credential)"
# An optional snake/kebab prefix keeps the full name in the redacted output,
# so ``GEMINI_API_KEY=...`` becomes ``GEMINI_API_KEY=[REDACTED]``.
_SENSITIVE_NAME = rf"(?:[A-Za-z0-9]+[_-])*{_SENSITIVE_KEYS}"
_ASSIGNMENT_PATTERN = re.compile(rf"(?i)\b({_SENSITIVE_NAME})(\s*[=:]\s*)\S+")
_AIZA_PATTERN = re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")
_BEARER_PATTERN = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/-]+=*")


def _redact_text(text: str) -> str:
    """Mask known sensitive patterns in *text*."""
    if not text:
        return text

    text = _ASSIGNMENT_PATTERN.sub(r"\1\2[REDACTED]", text)
    text = _AIZA_PATTERN.sub("[REDACTED]", text)
    text = _BEARER_PATTERN.sub(r"\1 [REDACTED]", text)
    return text


class RedactingFilter(logging.Filter):
    """Logging filter that redacts API keys, tokens, and passwords."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except (TypeError, ValueError, KeyError):
            msg = str(record.msg)

        record.msg = _redact_text(msg)
        record.args = ()

        if record.exc_text:
            record.exc_text = _redact_text(record.exc_text)

        return True


class JsonFormatter(logging.Formatter):
    """JSON Lines log formatter for file outputs."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.datetime.fromtimestamp(
            record.created, tz=datetime.UTC
        ).isoformat()

        log_data: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            log_data["exception"] = _redact_text(record.exc_text)

        if hasattr(record, "event") and isinstance(record.event, dict):
            log_data["event"] = record.event

        return json.dumps(log_data, ensure_ascii=False)


@dataclass(frozen=True)
class ModelCallEvent:
    """Recorded after each model API call."""

    model: str
    duration_ms: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    status: Literal["success", "error", "timeout", "rate_limited"]
    retry_count: int


@dataclass(frozen=True)
class ToolCallEvent:
    """Recorded after each tool execution."""

    tool_name: str
    duration_ms: float
    status: Literal["success", "denied", "timeout", "error"]


def emit_model_call(event: ModelCallEvent) -> None:
    """Emit a model call telemetry event if telemetry is enabled."""
    logger = logging.getLogger(TELEMETRY_LOGGER_NAME)
    if logger.isEnabledFor(logging.DEBUG):
        payload = asdict(event)
        logger.debug(
            "model_call: %s (%s)",
            event.model,
            event.status,
            extra={"event": payload, "event_type": "model_call"},
        )


def emit_tool_call(event: ToolCallEvent) -> None:
    """Emit a tool call telemetry event if telemetry is enabled."""
    logger = logging.getLogger(TELEMETRY_LOGGER_NAME)
    if logger.isEnabledFor(logging.DEBUG):
        payload = asdict(event)
        logger.debug(
            "tool_call: %s (%s)",
            event.tool_name,
            event.status,
            extra={"event": payload, "event_type": "tool_call"},
        )


def _default_log_dir() -> Path:
    try:
        return Path.home() / ".gemini-cli" / "logs"
    except RuntimeError:
        return Path("/tmp") / ".gemini-cli" / "logs"


def _cleanup_managed_handlers(logger: logging.Logger) -> None:
    """Close and remove handlers marked as managed by gemini_cli."""
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLERS_MARKER, False):
            logger.removeHandler(handler)
            handler.close()


def _parse_level(level: str | int) -> int:
    if isinstance(level, str):
        return int(getattr(logging, level.upper(), logging.INFO))
    return level


def configure_logging(
    level: str | int = logging.INFO,
    telemetry_enabled: bool = False,
    *,
    log_dir: Path | None = None,
) -> None:
    """Configure structured logging and optional telemetry.

    - Console output goes to stderr via RichHandler.
    - Application logs are saved as JSON Lines to ``<log_dir>/app.jsonl``.
    - Telemetry events (when enabled) are saved to ``<log_dir>/telemetry.jsonl``.

    When *telemetry_enabled* is ``False`` (or log files cannot be created), no
    telemetry handler is attached and the telemetry logger is disabled, so
    events are discarded before they reach any handler.
    """
    numeric_level = _parse_level(level)

    redacting_filter = RedactingFilter()

    main_logger = logging.getLogger(LOGGER_NAME)
    main_logger.setLevel(numeric_level)

    _cleanup_managed_handlers(main_logger)

    console_handler = RichHandler(
        console=Console(stderr=True),
        show_path=False,
        rich_tracebacks=True,
        markup=False,
    )
    console_handler.setLevel(numeric_level)
    console_handler.addFilter(redacting_filter)
    setattr(console_handler, _HANDLERS_MARKER, True)
    main_logger.addHandler(console_handler)

    resolved_dir = log_dir if log_dir is not None else _default_log_dir()
    file_handler_created = False

    try:
        resolved_dir.mkdir(parents=True, exist_ok=True)
        app_log_file = resolved_dir / "app.jsonl"
        app_file_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        app_file_handler.setLevel(numeric_level)
        app_file_handler.setFormatter(JsonFormatter())
        app_file_handler.addFilter(redacting_filter)
        setattr(app_file_handler, _HANDLERS_MARKER, True)
        main_logger.addHandler(app_file_handler)
        file_handler_created = True
    except OSError as exc:
        main_logger.warning(
            "Could not create log directory/file at %s: %s", resolved_dir, exc
        )

    telemetry_logger = logging.getLogger(TELEMETRY_LOGGER_NAME)
    telemetry_logger.setLevel(logging.DEBUG)
    telemetry_logger.propagate = False

    _cleanup_managed_handlers(telemetry_logger)
    telemetry_logger.disabled = not (telemetry_enabled and file_handler_created)

    if telemetry_logger.disabled:
        return

    telemetry_file = resolved_dir / "telemetry.jsonl"
    telemetry_handler = logging.handlers.RotatingFileHandler(
        telemetry_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    telemetry_handler.setLevel(logging.DEBUG)
    telemetry_handler.setFormatter(JsonFormatter())
    telemetry_handler.addFilter(redacting_filter)
    setattr(telemetry_handler, _HANDLERS_MARKER, True)
    telemetry_logger.addHandler(telemetry_handler)

"""Unit tests for telemetry and structured logging."""

from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path

import pytest
from rich.logging import RichHandler

from gemini_cli.telemetry.logger import (
    LOGGER_NAME,
    TELEMETRY_LOGGER_NAME,
    JsonFormatter,
    ModelCallEvent,
    RedactingFilter,
    ToolCallEvent,
    configure_logging,
    emit_model_call,
    emit_tool_call,
)


@pytest.fixture(autouse=True)
def _cleanup_logging():
    """Ensure loggers are cleaned up after each test."""
    yield
    for name in [LOGGER_NAME, TELEMETRY_LOGGER_NAME]:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        logger.disabled = False


def _record(message: str, *args: object) -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg=message,
        args=args,
        exc_info=None,
    )


def _telemetry_file_handlers() -> list[logging.Handler]:
    """Return telemetry file handlers, ignoring pytest's capture handlers."""
    logger = logging.getLogger(TELEMETRY_LOGGER_NAME)
    return [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.handlers.RotatingFileHandler)
        and Path(handler.baseFilename).name == "telemetry.jsonl"
    ]


def test_redacting_filter_assignments():
    filter_ = RedactingFilter()
    record = _record("Connecting with api_key=AIzaSySecret123 and password=supersecret")
    assert filter_.filter(record)
    assert "api_key=[REDACTED]" in record.msg
    assert "password=[REDACTED]" in record.msg
    assert "AIzaSySecret123" not in record.msg
    assert "supersecret" not in record.msg


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("api_key=AIzaSyB1234567890abcdefghij", "api_key=[REDACTED]"),
        ("GEMINI_API_KEY=abc123", "GEMINI_API_KEY=[REDACTED]"),
        ("token:abc", "token:[REDACTED]"),
        ("password = hunter2", "password = [REDACTED]"),
        ("monkey=banana", "monkey=banana"),
    ],
)
def test_redacting_filter_patterns(message: str, expected: str):
    filter_ = RedactingFilter()
    record = _record(message)

    assert filter_.filter(record)
    assert record.msg == expected


def test_redacting_filter_bare_aiza_key():
    filter_ = RedactingFilter()
    record = _record("Using key AIzaSyD3m0K3yStr1ngF0rT3st1ngA1B2C3D4")
    assert filter_.filter(record)
    assert "AIzaSyD3m0K3yStr1ngF0rT3st1ngA1B2C3D4" not in record.msg
    assert "[REDACTED]" in record.msg


def test_redacting_filter_args_formatting():
    filter_ = RedactingFilter()
    record = _record("Passed secret token=%s", "my_secret_token_value")
    assert filter_.filter(record)
    assert record.args == ()
    assert record.msg == "Passed secret token=[REDACTED]"
    assert "my_secret_token_value" not in record.msg


def test_redacting_filter_exc_text():
    filter_ = RedactingFilter()
    record = _record("Error occurred")
    record.exc_text = "Traceback: Exception with api_key=AIzaSyBSecretKey123"
    assert filter_.filter(record)
    assert "AIzaSyBSecretKey123" not in record.exc_text
    assert "api_key=[REDACTED]" in record.exc_text


def test_json_formatter_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="gemini_cli.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="Test log message",
        args=(),
        exc_info=None,
    )
    record.module = "test"
    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "gemini_cli.test"
    assert parsed["message"] == "Test log message"
    assert parsed["module"] == "test"
    assert parsed["line"] == 42
    assert "timestamp" in parsed


def test_json_formatter_redacts_exception_text():
    formatter = JsonFormatter()
    record = _record("Failure")
    record.exc_text = "Traceback: api_key=AIzaSyBSecretKey123"

    parsed = json.loads(formatter.format(record))

    assert "AIzaSyBSecretKey123" not in parsed["exception"]
    assert "api_key=[REDACTED]" in parsed["exception"]


def test_configure_logging_creates_handlers_and_app_log(tmp_path: Path):
    log_dir = tmp_path / "logs"
    configure_logging(level="DEBUG", telemetry_enabled=False, log_dir=log_dir)

    main_logger = logging.getLogger(LOGGER_NAME)
    assert len(main_logger.handlers) == 2  # Console + File
    assert any(isinstance(h, RichHandler) for h in main_logger.handlers)

    main_logger.info("Hello from gemini-cli")

    app_log = log_dir / "app.jsonl"
    assert app_log.exists()
    lines = app_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["message"] == "Hello from gemini-cli"


def test_configure_logging_sets_level(tmp_path: Path):
    log_dir = tmp_path / "logs"
    configure_logging(level="WARNING", telemetry_enabled=False, log_dir=log_dir)

    main_logger = logging.getLogger(LOGGER_NAME)
    assert main_logger.level == logging.WARNING
    for handler in main_logger.handlers:
        assert handler.level == logging.WARNING


def test_configure_logging_console_only_has_no_telemetry_handler(tmp_path: Path):
    log_dir = tmp_path / "logs"
    configure_logging("DEBUG", False, log_dir=log_dir)

    main_logger = logging.getLogger(LOGGER_NAME)
    assert any(isinstance(h, RichHandler) for h in main_logger.handlers)

    telemetry_logger = logging.getLogger(TELEMETRY_LOGGER_NAME)
    assert _telemetry_file_handlers() == []
    assert telemetry_logger.disabled is True
    assert not (log_dir / "telemetry.jsonl").exists()


def test_configure_logging_toggles_telemetry_handler(tmp_path: Path):
    log_dir = tmp_path / "logs"
    telemetry_logger = logging.getLogger(TELEMETRY_LOGGER_NAME)

    configure_logging("DEBUG", True, log_dir=log_dir)
    assert len(_telemetry_file_handlers()) == 1
    assert telemetry_logger.disabled is False

    configure_logging("DEBUG", False, log_dir=log_dir)
    assert _telemetry_file_handlers() == []
    assert telemetry_logger.disabled is True

    configure_logging("DEBUG", True, log_dir=log_dir)
    assert len(_telemetry_file_handlers()) == 1
    assert telemetry_logger.disabled is False


def test_console_output_redacts_api_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    log_dir = tmp_path / "logs"
    api_key = "AIzaSyB1234567890abcdefghijklmnop"
    configure_logging("DEBUG", False, log_dir=log_dir)

    logging.getLogger(LOGGER_NAME).info("Connecting with api_key=%s", api_key)

    stderr = capsys.readouterr().err
    assert "api_key=[REDACTED]" in stderr
    assert api_key not in stderr


def test_configure_logging_idempotent(tmp_path: Path):
    log_dir = tmp_path / "logs"
    configure_logging(level="INFO", telemetry_enabled=False, log_dir=log_dir)
    handlers_count_first = len(logging.getLogger(LOGGER_NAME).handlers)

    configure_logging(level="INFO", telemetry_enabled=False, log_dir=log_dir)
    handlers_count_second = len(logging.getLogger(LOGGER_NAME).handlers)

    assert handlers_count_first == handlers_count_second


def _model_event() -> ModelCallEvent:
    return ModelCallEvent(
        model="gemini-2.5-flash",
        duration_ms=120.5,
        input_tokens=100,
        output_tokens=50,
        cached_tokens=10,
        status="success",
        retry_count=0,
    )


def test_telemetry_disabled_suppresses_events(tmp_path: Path):
    log_dir = tmp_path / "logs"
    configure_logging(level="DEBUG", telemetry_enabled=False, log_dir=log_dir)

    emit_model_call(_model_event())

    assert not (log_dir / "telemetry.jsonl").exists()


def test_telemetry_disabled_events_do_not_leak_to_app_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    log_dir = tmp_path / "logs"
    configure_logging(level="DEBUG", telemetry_enabled=False, log_dir=log_dir)

    emit_model_call(_model_event())
    emit_tool_call(
        ToolCallEvent(tool_name="read_file", duration_ms=5.0, status="success")
    )

    app_log = log_dir / "app.jsonl"
    if app_log.exists():
        assert "model_call" not in app_log.read_text(encoding="utf-8")
    assert "model_call" not in capsys.readouterr().err


def test_telemetry_enabled_records_events(tmp_path: Path):
    log_dir = tmp_path / "logs"
    configure_logging(level="DEBUG", telemetry_enabled=True, log_dir=log_dir)

    model_event = ModelCallEvent(
        model="gemini-2.5-flash",
        duration_ms=150.0,
        input_tokens=200,
        output_tokens=80,
        cached_tokens=0,
        status="success",
        retry_count=1,
    )
    tool_event = ToolCallEvent(
        tool_name="read_file",
        duration_ms=25.0,
        status="success",
    )

    emit_model_call(model_event)
    emit_tool_call(tool_event)

    telemetry_log = log_dir / "telemetry.jsonl"
    assert telemetry_log.exists()

    lines = telemetry_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    parsed_model = json.loads(lines[0])
    assert parsed_model["event"]["model"] == "gemini-2.5-flash"
    assert parsed_model["event"]["input_tokens"] == 200

    parsed_tool = json.loads(lines[1])
    assert parsed_tool["event"]["tool_name"] == "read_file"
    assert parsed_tool["event"]["status"] == "success"


def test_telemetry_events_have_no_prompts_or_arguments():
    model_event = ModelCallEvent(
        model="gemini-2.5-pro",
        duration_ms=300.0,
        input_tokens=500,
        output_tokens=200,
        cached_tokens=50,
        status="success",
        retry_count=0,
    )
    tool_event = ToolCallEvent(
        tool_name="write_file",
        duration_ms=10.0,
        status="denied",
    )

    model_fields = set(model_event.__dataclass_fields__.keys())
    tool_fields = set(tool_event.__dataclass_fields__.keys())

    assert "prompt" not in model_fields
    assert "messages" not in model_fields
    assert "content" not in model_fields

    assert "arguments" not in tool_fields
    assert "args" not in tool_fields
    assert "output" not in tool_fields


def test_telemetry_payload_keys_match_event_fields(tmp_path: Path):
    log_dir = tmp_path / "logs"
    configure_logging(level="DEBUG", telemetry_enabled=True, log_dir=log_dir)

    emit_model_call(_model_event())
    emit_tool_call(
        ToolCallEvent(tool_name="read_file", duration_ms=5.0, status="success")
    )

    lines = (log_dir / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    model_payload, tool_payload = (json.loads(line)["event"] for line in lines)

    assert set(model_payload) == set(ModelCallEvent.__dataclass_fields__)
    assert set(tool_payload) == set(ToolCallEvent.__dataclass_fields__)

    forbidden = {"prompt", "messages", "content", "arguments", "args", "output"}
    assert not forbidden & set(model_payload)
    assert not forbidden & set(tool_payload)


def test_unwritable_directory_fallback(monkeypatch: pytest.MonkeyPatch):
    invalid_dir = Path("/proc/non_existent_directory_gemini_test/logs")
    configure_logging(level="INFO", telemetry_enabled=True, log_dir=invalid_dir)

    main_logger = logging.getLogger(LOGGER_NAME)
    assert len(main_logger.handlers) == 1

    telemetry_logger = logging.getLogger(TELEMETRY_LOGGER_NAME)
    assert _telemetry_file_handlers() == []
    assert telemetry_logger.disabled is True

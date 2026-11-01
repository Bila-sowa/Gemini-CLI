"""Tests for the tool registry."""

from __future__ import annotations

import dataclasses
import json
import logging

import pytest
from pydantic import BaseModel, ConfigDict

from gemini_cli.llm.models import ToolDefinition
from gemini_cli.telemetry import ToolCallEvent
from gemini_cli.tools import (
    BaseTool,
    DuplicateToolError,
    InvalidToolError,
    RiskLevel,
    ToolCategory,
    ToolNotFoundError,
    ToolRegistry,
    ToolResult,
)
from gemini_cli.tools import registry as registry_module


class EchoParams(BaseModel):
    """Parameters for the echo tool."""

    model_config = ConfigDict(extra="forbid")

    text: str
    times: int = 1


class SecretParams(BaseModel):
    """Parameters used to check that rejected input is not echoed."""

    api_key: str


class StubTool(BaseTool):
    """Configurable tool used to exercise registry behavior."""

    def __init__(
        self,
        name: str = "echo",
        description: str = "Echo text.",
        schema: type[BaseModel] = EchoParams,
        category: ToolCategory = ToolCategory.CODE,
        risk_level: RiskLevel = RiskLevel.READ,
    ) -> None:
        self._name = name
        self._description = description
        self._schema = schema
        self._category = category
        self._risk_level = risk_level

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> type[BaseModel]:
        return self._schema

    @property
    def category(self) -> ToolCategory:
        return self._category

    @property
    def risk_level(self) -> RiskLevel:
        return self._risk_level

    async def execute(self, params: BaseModel) -> ToolResult:
        return ToolResult(success=True, output="ok")


class EchoTool(StubTool):
    """Tool that echoes validated parameters."""

    async def execute(self, params: BaseModel) -> ToolResult:
        assert isinstance(params, EchoParams)
        output = params.text * params.times
        return ToolResult(success=True, output=output, metadata={"length": len(output)})


class RaisingTool(StubTool):
    """Tool whose execution always raises."""

    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc

    async def execute(self, params: BaseModel) -> ToolResult:
        raise self._exc


class BadResultTool(StubTool):
    """Tool that violates the return contract."""

    async def execute(self, params: BaseModel) -> ToolResult:
        return "not a result"  # type: ignore[return-value]


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[ToolCallEvent]:
    """Capture telemetry events instead of writing them to a handler."""
    captured: list[ToolCallEvent] = []
    monkeypatch.setattr(registry_module, "emit_tool_call", captured.append)
    return captured


def test_registry_starts_empty() -> None:
    registry = ToolRegistry()

    assert registry.list_tools() == []
    assert registry.export_schemas() == []


def test_register_and_get() -> None:
    registry = ToolRegistry()
    tool = StubTool()

    registry.register(tool)

    assert registry.get("echo") is tool


def test_duplicate_name_rejected() -> None:
    registry = ToolRegistry()
    registry.register(StubTool())

    with pytest.raises(DuplicateToolError, match="echo"):
        registry.register(StubTool())

    assert len(registry.list_tools()) == 1


def test_get_unknown_tool_reports_available_names() -> None:
    registry = ToolRegistry()
    registry.register(StubTool(name="alpha"))
    registry.register(StubTool(name="beta"))

    with pytest.raises(ToolNotFoundError) as excinfo:
        registry.get("missing")

    message = str(excinfo.value)
    assert "missing" in message
    assert "alpha" in message
    assert "beta" in message


def test_list_tools_preserves_order_and_returns_a_copy() -> None:
    registry = ToolRegistry()
    registry.register(StubTool(name="first"))
    registry.register(StubTool(name="second"))

    tools = registry.list_tools()
    tools.clear()

    assert [tool.name for tool in registry.list_tools()] == ["first", "second"]


def test_export_schemas_shape() -> None:
    registry = ToolRegistry()
    registry.register(StubTool())

    schemas = registry.export_schemas()

    assert len(schemas) == 1
    entry = schemas[0]
    assert set(entry) == {"name", "description", "parameters_json_schema"}
    assert entry["name"] == "echo"
    assert entry["description"] == "Echo text."
    schema = entry["parameters_json_schema"]
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"text", "times"}
    assert schema["required"] == ["text"]


def test_export_schemas_are_json_serializable() -> None:
    registry = ToolRegistry()
    registry.register(StubTool())

    payload = json.dumps(registry.export_schemas())

    assert '"name": "echo"' in payload


def test_export_schemas_round_trip_through_tool_definition() -> None:
    registry = ToolRegistry()
    registry.register(StubTool())

    definition = ToolDefinition.from_export(registry.export_schemas()[0])

    assert definition.name == "echo"
    assert definition.description == "Echo text."
    assert definition.parameters_json_schema["required"] == ["text"]


def test_register_accepts_dashed_names() -> None:
    registry = ToolRegistry()

    registry.register(StubTool(name="my-tool"))

    assert registry.get("my-tool").name == "my-tool"


@pytest.mark.parametrize("name", ["", "   ", "1bad", "bad name", "bad.name", "a" * 65])
def test_register_rejects_invalid_names(name: str) -> None:
    registry = ToolRegistry()

    with pytest.raises(InvalidToolError):
        registry.register(StubTool(name=name))


def test_register_rejects_empty_description() -> None:
    registry = ToolRegistry()

    with pytest.raises(InvalidToolError, match="description"):
        registry.register(StubTool(description="  "))


def test_register_rejects_non_model_schema() -> None:
    registry = ToolRegistry()

    with pytest.raises(InvalidToolError, match="parameters_schema"):
        registry.register(StubTool(schema=dict))  # type: ignore[arg-type]


async def test_dispatch_validates_and_executes() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    result = await registry.dispatch("echo", {"text": "ab", "times": 2})

    assert result.success is True
    assert result.output == "abab"
    assert result.metadata == {"length": 4}


async def test_dispatch_rejects_invalid_arguments_without_echoing_input() -> None:
    registry = ToolRegistry()
    registry.register(StubTool(schema=SecretParams))

    result = await registry.dispatch("echo", {"api_key": {"token": "SUPER_SECRET"}})

    assert result.success is False
    assert "api_key" in (result.error or "")
    assert "SUPER_SECRET" not in (result.error or "")


async def test_invalid_arguments_are_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = ToolRegistry()
    registry.register(StubTool(schema=SecretParams))

    with caplog.at_level(logging.WARNING, logger="gemini_cli.tools.registry"):
        await registry.dispatch("echo", {"api_key": {"token": "SUPER_SECRET"}})

    assert "SUPER_SECRET" not in caplog.text


async def test_dispatch_unknown_tool_returns_failure() -> None:
    registry = ToolRegistry()

    result = await registry.dispatch("missing", {})

    assert result.success is False
    assert "Unknown tool" in (result.error or "")


async def test_dispatch_handles_tool_exception() -> None:
    registry = ToolRegistry()
    registry.register(RaisingTool(RuntimeError("backend exploded")))

    result = await registry.dispatch("echo", {"text": "hi"})

    assert result.success is False
    assert "RuntimeError" in (result.error or "")
    assert "backend exploded" in (result.error or "")


async def test_dispatch_handles_timeout() -> None:
    registry = ToolRegistry()
    registry.register(RaisingTool(TimeoutError("slow")))

    result = await registry.dispatch("echo", {"text": "hi"})

    assert result.success is False
    assert "TimeoutError" in (result.error or "")


async def test_dispatch_rejects_invalid_result() -> None:
    registry = ToolRegistry()
    registry.register(BadResultTool())

    result = await registry.dispatch("echo", {"text": "hi"})

    assert result.success is False
    assert "invalid result" in (result.error or "")


async def test_dispatch_emits_success_event(events: list[ToolCallEvent]) -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    await registry.dispatch("echo", {"text": "hi"})

    assert len(events) == 1
    event = events[0]
    assert event.tool_name == "echo"
    assert event.status == "success"
    assert event.duration_ms >= 0
    assert {field.name for field in dataclasses.fields(event)} == {
        "tool_name",
        "duration_ms",
        "status",
    }


async def test_dispatch_emits_error_event_for_invalid_arguments(
    events: list[ToolCallEvent],
) -> None:
    registry = ToolRegistry()
    registry.register(StubTool())

    await registry.dispatch("echo", {})

    assert [event.status for event in events] == ["error"]


async def test_dispatch_emits_timeout_event(events: list[ToolCallEvent]) -> None:
    registry = ToolRegistry()
    registry.register(RaisingTool(TimeoutError("slow")))

    await registry.dispatch("echo", {"text": "hi"})

    assert [event.status for event in events] == ["timeout"]


async def test_unknown_tool_does_not_emit_event(events: list[ToolCallEvent]) -> None:
    registry = ToolRegistry()

    await registry.dispatch("missing", {})

    assert events == []

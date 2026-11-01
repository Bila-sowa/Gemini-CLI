"""Tests for the base tool contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from pydantic import BaseModel

from gemini_cli.tools import BaseTool, RiskLevel, ToolCategory, ToolResult


class EchoParams(BaseModel):
    """Parameters for the test tool."""

    text: str
    times: int = 1


class FakeTool(BaseTool):
    """Minimal concrete tool used to exercise the contract."""

    @property
    def name(self) -> str:
        return "fake_echo"

    @property
    def description(self) -> str:
        return "Echo text a number of times."

    @property
    def parameters_schema(self) -> type[BaseModel]:
        return EchoParams

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ

    async def execute(self, params: BaseModel) -> ToolResult:
        assert isinstance(params, EchoParams)
        return ToolResult(
            success=True,
            output=params.text * params.times,
            metadata={"length": len(params.text) * params.times},
        )


class FailingTool(FakeTool):
    """Tool that reports an expected failure without raising."""

    async def execute(self, params: BaseModel) -> ToolResult:
        return ToolResult(success=False, output="", error="operation denied")


def test_tool_category_values() -> None:
    assert ToolCategory.FILESYSTEM == "filesystem"
    assert ToolCategory.SHELL == "shell"
    assert ToolCategory.WEB == "web"
    assert ToolCategory.CODE == "code"
    assert ToolCategory.MCP == "mcp"
    assert isinstance(ToolCategory.MCP, str)


def test_risk_level_values() -> None:
    assert RiskLevel.READ == "read"
    assert RiskLevel.WRITE == "write"
    assert RiskLevel.DESTRUCTIVE == "destructive"
    assert isinstance(RiskLevel.DESTRUCTIVE, str)


def test_tool_result_defaults() -> None:
    result = ToolResult(success=True, output="done")

    assert result.success is True
    assert result.output == "done"
    assert result.error is None
    assert result.metadata == {}


def test_tool_result_metadata_is_not_shared() -> None:
    first = ToolResult(success=True, output="one")
    second = ToolResult(success=True, output="two")

    first.metadata["lines"] = 1

    assert second.metadata == {}


def test_tool_result_is_frozen() -> None:
    result = ToolResult(success=True, output="done")

    with pytest.raises(FrozenInstanceError):
        result.success = False  # type: ignore[misc]


def test_base_tool_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseTool()  # type: ignore[abstract]


def test_concrete_tool_exposes_contract() -> None:
    tool = FakeTool()

    assert tool.name == "fake_echo"
    assert tool.description
    assert tool.parameters_schema is EchoParams
    assert tool.category is ToolCategory.CODE
    assert tool.risk_level is RiskLevel.READ


async def test_execute_receives_validated_params() -> None:
    tool = FakeTool()
    params = tool.parameters_schema.model_validate({"text": "ab", "times": 2})

    result = await tool.execute(params)

    assert result.success is True
    assert result.output == "abab"
    assert result.metadata == {"length": 4}


async def test_execute_can_return_structured_failure() -> None:
    tool = FailingTool()

    result = await tool.execute(EchoParams(text="hello"))

    assert result.success is False
    assert result.error == "operation denied"


def test_parameters_schema_exports_json_schema() -> None:
    schema = FakeTool().parameters_schema.model_json_schema()

    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"text", "times"}
    assert schema["required"] == ["text"]

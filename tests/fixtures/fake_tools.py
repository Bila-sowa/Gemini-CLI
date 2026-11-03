"""Fake tools and approval callbacks for agent-loop tests."""

from __future__ import annotations

from pydantic import BaseModel

from gemini_cli.sandbox.approval import ApprovalDecision
from gemini_cli.tools.base import BaseTool, RiskLevel, ToolCategory, ToolResult


class FakeParams(BaseModel):
    """One optional string parameter, enough for dispatch tests."""

    value: str = ""


class FakeTool(BaseTool):
    """Tool that records calls and returns a pre-configured result."""

    def __init__(
        self,
        name: str = "fake_tool",
        *,
        result: ToolResult | None = None,
        risk_level: RiskLevel = RiskLevel.READ,
        category: ToolCategory = ToolCategory.CODE,
        description: str = "A fake tool used in tests.",
    ) -> None:
        self._name = name
        self._result = (
            result if result is not None else ToolResult(success=True, output="ok")
        )
        self._risk_level = risk_level
        self._category = category
        self._description = description
        self.calls: list[FakeParams] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> type[BaseModel]:
        return FakeParams

    @property
    def category(self) -> ToolCategory:
        return self._category

    @property
    def risk_level(self) -> RiskLevel:
        return self._risk_level

    async def execute(self, params: BaseModel) -> ToolResult:
        if not isinstance(params, FakeParams):
            return ToolResult(
                success=False,
                output="",
                error="Unexpected parameter type.",
            )
        self.calls.append(params)
        return self._result


class FakeApprovalCallback:
    """Approval callback that records prompts and returns a fixed decision."""

    def __init__(self, decision: ApprovalDecision = ApprovalDecision.APPROVED) -> None:
        self.decision = decision
        self.calls: list[tuple[str, str, str]] = []

    async def request_approval(
        self,
        action_description: str,
        risk_level: str,
        details: str,
    ) -> ApprovalDecision:
        self.calls.append((action_description, risk_level, details))
        return self.decision


__all__ = ["FakeApprovalCallback", "FakeParams", "FakeTool"]

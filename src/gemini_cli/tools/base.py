"""Base contracts for the agent tool system.

Every capability the agent can invoke — built-in or provided by an MCP server —
implements :class:`BaseTool`. The contract deliberately keeps tool code free of
provider SDK types: the registry validates arguments with the tool's Pydantic
schema and the LLM client translates exported declarations into its own
representation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class ToolCategory(StrEnum):
    """Capability group used for discovery, documentation, and UI grouping."""

    FILESYSTEM = "filesystem"
    SHELL = "shell"
    WEB = "web"
    CODE = "code"
    MCP = "mcp"


class RiskLevel(StrEnum):
    """Side-effect classification consumed by the approval layer.

    The string values are part of the approval contract, so the approval
    manager can compare them directly with the strings it receives.
    """

    READ = "read"
    """No side effects; safe to auto-approve."""

    WRITE = "write"
    """Creates or modifies resources; may require confirmation."""

    DESTRUCTIVE = "destructive"
    """Deletes resources or runs shell commands; always requires approval."""


@dataclass(frozen=True)
class ToolResult:
    """Structured result returned to the agent loop.

    Tools report *expected* failures — invalid input, missing files, denied
    operations — as ``ToolResult(success=False, error=...)`` rather than by
    raising. Unexpected exceptions may propagate to the registry, which turns
    them into a structured failure before the agent sees them.

    ``metadata`` is for safe, structured extras such as line counts or HTTP
    status codes. It must never contain secrets, raw prompts, or unredacted
    tool arguments.
    """

    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Contract that every tool must implement.

    Implementations should be stateless with respect to a single call: all
    per-call input arrives through validated ``execute`` parameters.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name used in model declarations and dispatch."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description included in model tool declarations."""

    @property
    @abstractmethod
    def parameters_schema(self) -> type[BaseModel]:
        """Pydantic model that validates and documents tool parameters.

        The registry uses this model both to validate model-generated arguments
        and to export a JSON Schema declaration for the LLM.
        """

    @property
    @abstractmethod
    def category(self) -> ToolCategory:
        """Capability group the tool belongs to."""

    @property
    @abstractmethod
    def risk_level(self) -> RiskLevel:
        """Side-effect classification used by the approval layer."""

    @abstractmethod
    async def execute(self, params: BaseModel) -> ToolResult:
        """Run the tool with validated parameters and return a result.

        ``params`` is an instance of :attr:`parameters_schema`, validated by
        the registry. Implementations must not perform side effects before
        validating the concrete parameter type.
        """


__all__ = ["BaseTool", "RiskLevel", "ToolCategory", "ToolResult"]
